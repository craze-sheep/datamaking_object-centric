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

# === PATH SETUP ===
# ai_model/ must come before model/ in sys.path so "from config import" finds ai_model/config.py
_this_dir = os.path.dirname(os.path.abspath(__file__))          # model/ai_model
_model_dir = os.path.dirname(_this_dir)                          # model
_project_root = os.path.dirname(_model_dir)                      # project root

# model/ first, then model/ai_model/ on top (higher priority)
# This way "from config import" finds ai_model/config.py, not model/config.py
# But data_adapter.py can still access model/dataset.py via importlib by path
sys.path.insert(0, _model_dir)
sys.path.insert(0, _this_dir)

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from config import AIModelConfig
from data_adapter import create_dataloader
from model import PhysicsObjectGraphPredictor


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def try_forward_backward(model, batch, optimizer, scaler, device, use_amp):
    """Single forward-backward step. Returns loss dict or raises on OOM."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    if use_amp:
        with autocast():
            loss_dict = model.forward_with_loss(batch_dev)
        scaler.scale(loss_dict['total_loss']).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_dict = model.forward_with_loss(batch_dev)
        loss_dict['total_loss'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    config.state_decoder_hidden = max(64, config.state_decoder_hidden // 2)
    config.collision_hidden = max(16, config.collision_hidden // 2)
    config.mask_base_channels = max(8, config.mask_base_channels // 2)
    config.rgb_base_channels = max(8, config.rgb_base_channels // 2)
    config.gnn_layers = max(1, config.gnn_layers - 1)
    config.gru_num_layers = max(1, config.gru_num_layers - 1)
    return config


def find_working_config(config, database_root, device):
    """Find batch_size and model config that fit in GPU memory.

    Strategy: start from given batch_size, halve on OOM. If bs=1 OOM, shrink model.
    """
    batch_size = config.batch_size
    use_amp = config.use_amp and device.type == 'cuda'
    model_shrunk = False

    while True:
        print(f"\n  Trying batch_size={batch_size}, fused_dim={config.fused_dim}...")
        try:
            loader = create_dataloader(
                database_root, 'train', batch_size,
                config.history_length, config.predict_length,
                config.max_objects, num_workers=0,
            )
            batch = next(iter(loader))

            model = PhysicsObjectGraphPredictor(config).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = GradScaler() if use_amp else None

            loss_dict = try_forward_backward(model, batch, optimizer, scaler, device, use_amp)
            print(f"  SUCCESS! loss={loss_dict['total_loss']:.4f}")

            del model, optimizer, scaler
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            return batch_size, config, use_amp, model_shrunk

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

            if batch_size > 1:
                batch_size //= 2
            elif not model_shrunk:
                print("  Shrinking model...")
                config = shrink_model(config)
                model_shrunk = True
            else:
                raise RuntimeError("Cannot fit model in GPU memory even with smallest config")


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
                    ld = model.forward_with_loss(batch_dev)
            else:
                ld = model.forward_with_loss(batch_dev)
            for k in ('total_loss', 'rgb_loss', 'state_loss', 'collision_loss', 'mask_loss'):
                sums[k] = sums.get(k, 0) + ld[k].item()
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

    database_root = os.path.join(_project_root, 'database')

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
        config.num_epochs = args.epochs or 5
        max_steps = args.max_steps or 50
        num_epochs = config.num_epochs

    print(f"\nConfig: fused_dim={config.fused_dim}, gru={config.gru_hidden_dim}, "
          f"gnn={config.gnn_hidden_dim}, history={config.history_length}, "
          f"predict={config.predict_length}, max_obj={config.max_objects}")

    # Find working config
    print("\n=== Auto-tuning batch size ===")
    batch_size, config, use_amp, model_shrunk = find_working_config(
        config, database_root, device
    )
    config.batch_size = batch_size

    # Build model
    print(f"\n=== Building model (batch_size={batch_size}, amp={use_amp}) ===")
    model = PhysicsObjectGraphPredictor(config).to(device)
    total_p, train_p = count_params(model)
    print(f"Parameters: {total_p:,} total, {train_p:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                   weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler() if use_amp else None

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val = float('inf')
    if args.resume and os.path.exists(args.resume):
        print(f"\n=== Resuming from {args.resume} ===")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_val = ckpt.get('best_val_loss', float('inf'))
        print(f"  Resumed from epoch {start_epoch}, best_val_loss={best_val:.4f}")

    # Data
    print("\n=== Loading data ===")
    train_loader = create_dataloader(
        database_root, 'train', batch_size,
        config.history_length, config.predict_length,
        config.max_objects, num_workers=0,
    )
    val_loader = create_dataloader(
        database_root, 'val', batch_size,
        config.history_length, config.predict_length,
        config.max_objects, num_workers=0,
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Checkpoint dir
    ckpt_dir = os.path.join(_this_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    print(f"\n=== Training: {num_epochs} epochs x {max_steps} steps ===")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        losses = {'total': 0, 'rgb': 0, 'state': 0, 'collision': 0, 'mask': 0}
        n_steps = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            if step >= max_steps:
                break
            try:
                ld = try_forward_backward(model, batch, optimizer, scaler, device, use_amp)
                for k in losses:
                    losses[k] += ld.get(f'{k}_loss', ld.get('total_loss', 0))
                n_steps += 1

                if (step + 1) % max(1, max_steps // 4) == 0 or step == 0:
                    print(f"  [{step+1}/{max_steps}] loss={ld['total_loss']:.4f} "
                          f"rgb={ld['rgb_loss']:.4f} state={ld['state_loss']:.4f} "
                          f"coll={ld['collision_loss']:.4f} mask={ld['mask_loss']:.4f}")

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

        # Validate
        vl = validate(model, val_loader, device, use_amp)
        print(f"  Val:   total={vl['total_loss']:.4f} rgb={vl['rgb_loss']:.4f} "
              f"state={vl['state_loss']:.4f} coll={vl['collision_loss']:.4f} "
              f"mask={vl['mask_loss']:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses['total'],
            'config': config,
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

        if vl['total_loss'] < best_val:
            best_val = vl['total_loss']
            best_path = os.path.join(ckpt_dir, 'best.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'config': config,
                'best_val_loss': best_val,
            }, best_path)
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


def main():
    parser = argparse.ArgumentParser(description='Train PhysicsObjectGraphPredictor')
    parser.add_argument('--mode', choices=['smoke', 'small', 'train'], default='smoke',
                        help='smoke: 3 steps, small: ~20 steps, train: full')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pt file to resume from')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
