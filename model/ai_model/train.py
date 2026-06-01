"""Training script for PhysicsObjectGraphPredictor.

Features:
  - Auto batch size reduction on CUDA OOM (halves until batch_size=1)
  - If batch_size=1 still OOM, reduces model size
  - AMP (mixed precision) for memory efficiency
  - Gradient clipping
  - Checkpoint saving
  - Loss logging
  - Validation loop

Usage:
  conda run -n model python model/ai_model/train.py --mode smoke
  conda run -n model python model/ai_model/train.py --mode train --epochs 5
"""
import sys
import os
import argparse
import time
import json
import gc
from dataclasses import asdict, fields, is_dataclass

# === PATH SETUP ===
# ai_model/ must come before model/ in sys.path so "from config import" finds ai_model/config.py
_this_dir = os.path.dirname(os.path.abspath(__file__))          # model/ai_model
_model_dir = os.path.dirname(_this_dir)                          # model
_project_root = os.path.dirname(_model_dir)                      # project root

# model/ first, then model/ai_model/ on top (higher priority)
# This way "from config import" finds ai_model/config.py, not model/config.py
# data_adapter.py loads dataset.py from the same directory via importlib
sys.path.insert(0, _model_dir)
sys.path.insert(0, _this_dir)

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from config import AIModelConfig
from data_adapter import create_dataloader
from model import PhysicsObjectGraphPredictor

LOSS_LOG_KEYS = ('total', 'rgb', 'ssim', 'lpips', 'state', 'collision', 'effect', 'mask', 'energy')
TB_PREFIX_NAMES = {
    'train_step': '训练',
    'train_epoch': '训练',
    'val_epoch': '验证',
}
TB_TAG_NAMES = {
    'total_loss': '01 总损失',
    'rgb_loss': '02 图像重建损失',
    'ssim_loss': '03 结构相似度损失',
    'lpips_loss': '04 感知相似度损失',
    'state_loss': '05 物理状态损失',
    'collision_loss': '06 碰撞分类损失',
    'effect_loss': '07 碰撞效果损失',
    'mask_loss': '08 分割掩码损失',
    'energy_loss': '09 能量守恒损失',
}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def create_summary_writer(args):
    """Create an optional TensorBoard writer for scalar training curves."""
    if args.no_tensorboard:
        return None
    if SummaryWriter is None:
        print("TensorBoard disabled: install `tensorboard` to enable SummaryWriter.")
        return None
    run_name = f"ai_model_{args.mode}_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = args.log_dir or os.path.join(_model_dir, 'runs', run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logdir: {log_dir}")
    print(f"View with: tensorboard --logdir {log_dir} --port 6006")
    print("Local URL: http://localhost:6006")
    return writer


def resolve_data_roots(args):
    """Resolve train/validation roots and split modes."""
    default_root = os.path.join(_project_root, 'database')
    train_root = os.path.abspath(args.train_root) if args.train_root else default_root
    val_root = os.path.abspath(args.val_root) if args.val_root else default_root
    explicit_roots = args.train_root is not None or args.val_root is not None
    train_split = 'all' if args.train_root else 'train'
    val_split = 'all' if explicit_roots else 'val'
    return train_root, val_root, train_split, val_split


def log_scalars(writer, prefix, values, step):
    """Write loss scalars only, keeping TensorBoard focused."""
    if writer is None:
        return
    prefix_name = TB_PREFIX_NAMES.get(prefix, prefix)
    for key, value in values.items():
        if not key.endswith('_loss'):
            continue
        if isinstance(value, (int, float)):
            scalar = value
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            scalar = value.item()
        else:
            continue
        tag_name = TB_TAG_NAMES.get(key, key)
        writer.add_scalar(f'{prefix_name}/{tag_name}', scalar, step)
    writer.flush()


def config_from_checkpoint(ckpt, fallback_config):
    """Build AIModelConfig from new dict checkpoints or older dataclass checkpoints."""
    ckpt_config = ckpt.get('config') if isinstance(ckpt, dict) else None
    if ckpt_config is None:
        return fallback_config

    if isinstance(ckpt_config, AIModelConfig):
        config_dict = vars(ckpt_config)
    elif is_dataclass(ckpt_config):
        config_dict = asdict(ckpt_config)
    elif isinstance(ckpt_config, dict):
        config_dict = ckpt_config
    else:
        return fallback_config

    allowed = {f.name for f in fields(AIModelConfig)}
    merged = asdict(AIModelConfig())
    merged.update({k: v for k, v in config_dict.items() if k in allowed})
    return AIModelConfig(**merged)


def checkpoint_state(epoch, model, optimizer, scheduler, scaler, losses, config, best_val):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': losses['total'],
        'config': asdict(config),
        'best_val_loss': best_val,
    }
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
    return state


def scheduled_sampling_ratio(config, epoch, total_epochs):
    """Linear teacher forcing decay for GRU scheduled sampling."""
    if not config.use_scheduled_sampling or total_epochs <= 1:
        return config.scheduled_sampling_end
    progress = epoch / max(total_epochs - 1, 1)
    return (
        config.scheduled_sampling_start
        + (config.scheduled_sampling_end - config.scheduled_sampling_start) * progress
    )


def try_forward_backward(
    model,
    batch,
    optimizer,
    scaler,
    device,
    use_amp,
    teacher_ratio=0.0,
    grad_scale=1.0,
    do_step=True,
):
    """Single forward-backward step. Returns loss dict or raises on OOM."""
    model.train()
    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    if use_amp:
        with autocast():
            loss_dict = model.forward_with_loss(batch_dev, teacher_ratio=teacher_ratio)
            loss = loss_dict['total_loss'] / grad_scale
        scaler.scale(loss).backward()
        if do_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), model.config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
    else:
        loss_dict = model.forward_with_loss(batch_dev, teacher_ratio=teacher_ratio)
        (loss_dict['total_loss'] / grad_scale).backward()
        if do_step:
            nn.utils.clip_grad_norm_(model.parameters(), model.config.gradient_clip)
            optimizer.step()

    return {k: (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in loss_dict.items() if k != 'pred'}


def shrink_model(config):
    """Halve model dimensions for lower memory."""
    config.cnn_channels = tuple(max(16, c // 2) for c in config.cnn_channels)
    config.visual_out_dim = max(32, config.visual_out_dim // 2)
    config.physics_out_dim = max(32, config.physics_out_dim // 2)
    config.fused_dim = max(32, config.fused_dim // 2)
    config.gnn_hidden_dim = max(32, config.gnn_hidden_dim // 2)
    config.gnn_edge_dim = max(16, config.gnn_edge_dim // 2)
    config.gru_hidden_dim = max(32, config.gru_hidden_dim // 2)
    config.temporal_attention_heads = min(config.temporal_attention_heads, config.gru_hidden_dim)
    config.state_decoder_hidden = max(64, config.state_decoder_hidden // 2)
    config.collision_hidden = max(16, config.collision_hidden // 2)
    config.mask_base_channels = max(8, config.mask_base_channels // 2)
    config.rgb_base_channels = max(8, config.rgb_base_channels // 2)
    config.gnn_layers = max(1, config.gnn_layers - 1)
    config.gru_num_layers = max(1, config.gru_num_layers - 1)
    return config


def _cuda_memory_fraction(device):
    if device.type != 'cuda':
        return 0.0
    used = torch.cuda.max_memory_allocated(device)
    props = torch.cuda.get_device_properties(device)
    return used / max(props.total_memory, 1)


def find_working_config(config, train_root, train_split, device, target_vram_fraction, max_batch_size):
    """Find batch_size and model config that fit in GPU memory.

    Strategy: probe increasing batch sizes and keep the largest one under
    target_vram_fraction. On CPU or resume, this is skipped by the caller.
    """
    use_amp = config.use_amp and device.type == 'cuda'
    model_shrunk = False
    batch_size = max(1, config.batch_size)
    best_batch = None
    best_fraction = 0.0
    target_vram_fraction = max(0.1, min(target_vram_fraction, 0.95))

    while True:
        print(f"\n  Trying batch_size={batch_size}, fused_dim={config.fused_dim}...")
        try:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            loader = create_dataloader(
                train_root, train_split, batch_size,
                config.history_length, config.predict_length,
                config.max_objects, num_workers=0,
            )
            batch = next(iter(loader))

            model = PhysicsObjectGraphPredictor(config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = GradScaler() if use_amp else None

            optimizer.zero_grad(set_to_none=True)
            loss_dict = try_forward_backward(model, batch, optimizer, scaler, device, use_amp, do_step=True)
            mem_fraction = _cuda_memory_fraction(device)
            print(f"  SUCCESS! loss={loss_dict['total_loss']:.4f}, peak_vram={mem_fraction:.1%}")

            del model, optimizer, scaler
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            if device.type != 'cuda':
                return batch_size, config, use_amp, model_shrunk

            if mem_fraction <= target_vram_fraction:
                best_batch = batch_size
                best_fraction = mem_fraction
                next_batch = batch_size * 2
                if next_batch > max_batch_size:
                    break
                batch_size = next_batch
                continue
            break

        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            print(f"  OOM: {e}")
            try:
                del model, optimizer, scaler
            except NameError:
                pass
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            if best_batch is not None:
                break
            if batch_size > 1:
                batch_size = max(1, batch_size // 2)
            elif not model_shrunk:
                print("  Shrinking model...")
                config = shrink_model(config)
                model_shrunk = True
            else:
                raise RuntimeError("Cannot fit model in GPU memory even with smallest config")

    if best_batch is None:
        best_batch = max(1, batch_size // 2)
    print(f"  Selected batch_size={best_batch} (target_vram<={target_vram_fraction:.0%}, observed={best_fraction:.1%})")
    return best_batch, config, use_amp, model_shrunk


def validate(model, val_loader, device, use_amp):
    """Run validation, return average losses."""
    model.eval()
    sums = {}
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            if use_amp:
                with autocast():
                    ld = model.forward_with_loss(batch_dev, teacher_ratio=0.0)
            else:
                ld = model.forward_with_loss(batch_dev, teacher_ratio=0.0)
            for k, v in ld.items():
                if k == 'pred' or not k.endswith('_loss'):
                    continue
                sums[k] = sums.get(k, 0) + v.item()
            n += 1
            if n >= 10:  # limit validation batches
                break
    if n == 0:
        return {k: 0.0 for k in sums}
    return {k: v / n for k, v in sums.items()}


def train(args):
    device = get_device()
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    train_root, val_root, train_split, val_split = resolve_data_roots(args)
    print(f"Train root: {train_root} (split={train_split})")
    print(f"Val root:   {val_root} (split={val_split})")

    # Config
    if args.mode == 'smoke':
        config = AIModelConfig.tiny()
        config.batch_size = 2
        config.use_amp = False  # simpler debugging
        max_steps = 3
        num_epochs = 1
    elif args.mode == 'small':
        config = AIModelConfig.small_8gb()
        config.batch_size = 4
        max_steps = 20
        num_epochs = 2
    else:
        config = AIModelConfig.small_8gb()
        config.batch_size = args.batch_size or 4
        config.num_epochs = args.epochs or config.num_epochs
        max_steps = args.max_steps
        num_epochs = config.num_epochs
    if args.max_steps is not None:
        max_steps = args.max_steps

    resume_ckpt = None
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f"\n=== Loading resume checkpoint metadata: {args.resume} ===")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        config = config_from_checkpoint(resume_ckpt, config)
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.mode == 'smoke':
            config.use_amp = False
            max_steps = args.max_steps or 3
            num_epochs = args.epochs or 1
        elif args.mode == 'small':
            max_steps = args.max_steps or 20
            num_epochs = args.epochs or max(2, resume_ckpt.get('epoch', 0))
        else:
            max_steps = args.max_steps
            num_epochs = args.epochs or config.num_epochs
        config.num_epochs = num_epochs

    print(f"\nConfig: fused_dim={config.fused_dim}, gru={config.gru_hidden_dim}, "
          f"gnn={config.gnn_hidden_dim}, history={config.history_length}, "
          f"predict={config.predict_length}, max_obj={config.max_objects}")

    if resume_ckpt is None:
        # Find working config
        print("\n=== Auto-tuning batch size ===")
        batch_size, config, use_amp, model_shrunk = find_working_config(
            config,
            train_root,
            train_split,
            device,
            args.target_vram_fraction,
            args.max_batch_size,
        )
        config.batch_size = batch_size
    else:
        batch_size = config.batch_size
        use_amp = config.use_amp and device.type == 'cuda'
        model_shrunk = False
        print("\n=== Skipping auto-tune for resume; using checkpoint config ===")

    # Build model
    print(f"\n=== Building model (batch_size={batch_size}, amp={use_amp}) ===")
    model = PhysicsObjectGraphPredictor(config).to(device)
    total_p, train_p = count_params(model)
    print(f"Parameters: {total_p:,} total, {train_p:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                   weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler() if use_amp else None
    writer = create_summary_writer(args)
    if writer is not None:
        writer.add_text('运行配置/参数', json.dumps(asdict(config), indent=2), 0)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val = float('inf')
    if resume_ckpt is not None:
        print(f"\n=== Resuming from {args.resume} ===")
        try:
            missing, unexpected = model.load_state_dict(
                resume_ckpt['model_state_dict'], strict=False
            )
        except RuntimeError as e:
            raise RuntimeError(
                "Checkpoint model weights are incompatible with the resolved config. "
                "Use the matching checkpoint config or start a fresh run."
            ) from e
        if missing:
            print(f"  Missing keys initialized from current model: {len(missing)}")
        if unexpected:
            print(f"  Unexpected checkpoint keys ignored: {len(unexpected)}")
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if resume_ckpt.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        if scaler is not None and resume_ckpt.get('scaler_state_dict') is not None:
            scaler.load_state_dict(resume_ckpt['scaler_state_dict'])
        start_epoch = resume_ckpt.get('epoch', 0)
        best_val = resume_ckpt.get('best_val_loss', float('inf'))
        print(f"  Resumed from epoch {start_epoch}, best_val_loss={best_val:.4f}")

    # Data
    print("\n=== Loading data ===")
    train_loader = create_dataloader(
        train_root, train_split, batch_size,
        config.history_length, config.predict_length,
        config.max_objects, num_workers=0,
    )
    val_loader = create_dataloader(
        val_root, val_split, batch_size,
        config.history_length, config.predict_length,
        config.max_objects, num_workers=0, shuffle=False, drop_last=False,
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Checkpoint dir
    ckpt_dir = os.path.abspath(args.checkpoint_dir) if args.checkpoint_dir else os.path.join(_this_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints dir: {ckpt_dir}")

    # Training loop
    steps_label = 'full epoch' if max_steps is None else f'{max_steps} steps'
    print(f"\n=== Training: {num_epochs} epochs x {steps_label} ===")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        losses = {k: 0 for k in LOSS_LOG_KEYS}
        n_steps = 0
        t0 = time.time()
        teacher_ratio = scheduled_sampling_ratio(config, epoch, num_epochs)
        accum_steps = max(1, config.gradient_accumulation_steps)
        last_step = len(train_loader) if max_steps is None else min(max_steps, len(train_loader))
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            if max_steps is not None and step >= max_steps:
                break
            try:
                do_step = ((step + 1) % accum_steps == 0) or ((step + 1) >= last_step)
                ld = try_forward_backward(
                    model,
                    batch,
                    optimizer,
                    scaler,
                    device,
                    use_amp,
                    teacher_ratio=teacher_ratio,
                    grad_scale=accum_steps,
                    do_step=do_step,
                )
                if do_step:
                    optimizer.zero_grad(set_to_none=True)
                for k in losses:
                    losses[k] += ld.get(f'{k}_loss', ld.get('total_loss', 0))
                n_steps += 1
                global_step = epoch * len(train_loader) + step + 1
                if step == 0 or (step + 1) % args.tb_log_every == 0 or (step + 1) >= last_step:
                    log_scalars(writer, 'train_step', ld, global_step)
                progress_total = last_step if max_steps is None else max_steps
                log_every = max(1, progress_total // 4)
                if (step + 1) % log_every == 0 or step == 0:
                    print(f"  [{step+1}/{progress_total}] loss={ld['total_loss']:.4f} "
                          f"rgb={ld['rgb_loss']:.4f} state={ld['state_loss']:.4f} "
                          f"coll={ld['collision_loss']:.4f} mask={ld['mask_loss']:.4f} "
                          f"teacher={teacher_ratio:.2f}")

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  OOM at step {step}, skipping")
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                raise

        scheduler.step()
        elapsed = time.time() - t0
        for k in losses:
            losses[k] /= max(n_steps, 1)

        print(f"\nEpoch {epoch+1}/{num_epochs} ({elapsed:.0f}s):")
        print(f"  Train: total={losses['total']:.4f} rgb={losses['rgb']:.4f} "
              f"state={losses['state']:.4f} coll={losses['collision']:.4f} "
              f"mask={losses['mask']:.4f}")
        print(f"         aux: energy={losses['energy']:.4f} effect={losses['effect']:.4f} "
              f"lpips={losses['lpips']:.4f}")
        epoch_step = (epoch + 1) * last_step
        log_scalars(
            writer,
            'train_epoch',
            {f'{k}_loss': v for k, v in losses.items()},
            epoch_step,
        )

        # Validate
        vl = validate(model, val_loader, device, use_amp)
        print(f"  Val:   total={vl['total_loss']:.4f} rgb={vl['rgb_loss']:.4f} "
              f"state={vl['state_loss']:.4f} coll={vl['collision_loss']:.4f} "
              f"mask={vl['mask_loss']:.4f}")
        print(f"         aux: energy={vl.get('energy_loss', 0.0):.4f} "
              f"effect={vl.get('effect_loss', 0.0):.4f} "
              f"lpips={vl.get('lpips_loss', 0.0):.4f}")
        log_scalars(writer, 'val_epoch', vl, epoch_step)

        improved = vl['total_loss'] < best_val
        if improved:
            best_val = vl['total_loss']

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch+1}.pt')
        state = checkpoint_state(
            epoch + 1, model, optimizer, scheduler, scaler, losses, config, best_val
        )
        torch.save(state, ckpt_path)
        print(f"  Saved: {ckpt_path}")

        if improved:
            best_path = os.path.join(ckpt_dir, 'best.pt')
            torch.save(state, best_path)
            print(f"  Best model saved: {best_path}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    print(f"  Batch size: {batch_size}")
    print(f"  AMP: {use_amp}")
    print(f"  Model shrunk: {model_shrunk}")
    print(f"  Parameters: {total_p:,}")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  History: {config.history_length}, Predict: {config.predict_length}")
    print(f"  Dims: fused={config.fused_dim}, gru={config.gru_hidden_dim}, gnn={config.gnn_hidden_dim}")
    if writer is not None:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train PhysicsObjectGraphPredictor')
    parser.add_argument('--mode', choices=['smoke', 'small', 'train'], default='smoke',
                        help='smoke: 3 steps, small: ~20 steps, train: full')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--train-root', type=str, default=None,
                        help='Explicit training data root; uses all samples under it')
    parser.add_argument('--val-root', type=str, default=None,
                        help='Explicit validation/test data root; uses all samples under it')
    parser.add_argument('--target-vram-fraction', type=float, default=0.82,
                        help='Auto batch target peak CUDA memory fraction')
    parser.add_argument('--max-batch-size', type=int, default=64,
                        help='Upper bound for auto batch probing')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pt file to resume from')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory for epoch checkpoints and best.pt')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='TensorBoard log directory; defaults to model/runs/<run_name>')
    parser.add_argument('--tb-log-every', type=int, default=20,
                        help='Write training loss scalars every N steps')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable TensorBoard SummaryWriter logging')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
