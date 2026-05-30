# Experiment Report: PhysicsObjectGraphPredictor

## 1. Model Design

Physics-aware object-centric video prediction with:
- Dual-stream encoder (visual CNN + physics embedding)
- Force-aware GNN interaction (force matrix as graph edges)
- GRU temporal prediction (autoregressive)
- Multi-head decoder (state, collision, mask, RGB)

Key difference from baseline PhysicsVideoPredictor: GRU instead of Transformer, explicit GNN with force edges.

## 2. Data Fields Used

| Field | Source | Used For |
|-------|--------|----------|
| RGB frames | dynamic/*/*.png | Visual encoder input, RGB loss target |
| Object masks | object_segment/*.npz | ROI pooling, mask loss target |
| Static attrs | object_static.json | Physics encoder (mass, friction, geometry, type) |
| Dynamic state | object_dynamicjson/*.json | Physics encoder + GNN edge features |
| Force matrix | force_matrix.json | GNN edge features, collision labels |
| Video metadata | video.json | Data loading (not direct model input) |

## 3. Data Fields NOT Used

| Field | Reason | Future Plan |
|-------|--------|-------------|
| Depth map *.npz | Per task requirement | Add as supervision signal |
| Camera params | Single-view, fixed | Multi-view fusion module |
| angular_velocity | Encoded in state but not separately modeled | Rotation-aware decoder |
| subtask/main_variable | String fields, unencoded | Text embedding injection |

## 4. Training Results

### Environment
- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM), CUDA 12.1, PyTorch 2.5.1

### Smoke Test (tiny config, real data)
- Config: fused_dim=32, gru=32, history=4, predict=4, max_obj=3
- Batch size: 2, Parameters: 132,032
- 22,315 train samples, 4,783 val samples
- 3 steps, loss: 13809 → 13642

### Small Training (medium config, 50 steps)
- Config: fused_dim=64, gru=64, history=6, predict=6, max_obj=3
- Batch size: 4, Parameters: 363,904
- Loss weights: rgb=1.0, state=1.0, coll=1.0, mask=0.001

| Metric | Step 1 | Step 50 | Note |
|--------|--------|---------|------|
| total | 71.46 | 60.66 | ↓15% |
| state | 6.55 | 0.12 | Rapid drop due to normalized targets |
| collision | 0.90 | 0.03 | Learning to predict no-collision (dominant class) |
| rgb | 1.23 | 0.45 | ↓64% |

### Caveats
- 50 steps is NOT convergence - it's a smoke test proving the pipeline works
- State loss drop is fast because targets are normalized (std~5-50)
- Collision loss drop is partly because most pairs are non-collision
- OOM auto-reduction was implemented but not triggered (peak 205MB/8GB)

### Verified
- Forward + Loss + Backward + Optimizer step complete on real data
- GPU used (RTX 4060 Laptop, CUDA 12.1)
- No NaN/Inf in loss
- Checkpoint saved successfully
- OOM auto batch-size reduction implemented

## 5. Checkpoints

- model/ai_model/checkpoints/best.pt
- model/ai_model/checkpoints/balanced_train.pt

## 6. Known Issues

1. Mask loss raw value ~40K-60K (pixel-level BCE), masked by weight 0.001
2. No checkpoint resume (--resume flag missing)
3. attr_dim=14 but only 13 values populated
4. No DINOv2 pretrained visual features

## 7. Next Steps

1. Longer training (1000+ steps) to verify real convergence
2. Add resume functionality
3. Add evaluation metrics (PSNR, SSIM, state MSE, collision F1)
4. Scale model to 1M+ parameters
5. Add DINOv2 visual backbone
