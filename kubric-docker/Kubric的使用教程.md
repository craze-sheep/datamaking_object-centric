# Kubric 使用教程

## 环境概览

| 组件 | 版本 | 说明 |
|------|------|------|
| Blender | 3.6.15 LTS | 渲染引擎，支持 RTX 4060 GPU |
| Kubric | 2022.4.1 | 物理仿真数据集生成框架 |
| PyBullet | latest | 物理引擎（CPU） |
| CUDA | 12.2 (容器) / 12.6 (驱动) | GPU 渲染加速 |
| Docker 镜像 | `kubric-gpu` | 基于 nvidia/cuda:12.2.0-base-ubuntu22.04 |

## 前置条件

1. Windows 主机安装 NVIDIA 驱动（≥560）
2. WSL2 中 Docker 已配置 GPU：
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
3. 验证：
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

## 构建镜像

```bash
cd /home/lzy/project/slot-datamaking/kubric-docker
docker build -f Dockerfile.kubric-gpu -t kubric-gpu .
```

## 运行数据生成

基本格式：

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  --volume "/home/lzy/project/slot-datamaking:/workspace" \
  --volume "/home/lzy/project/slot-datamaking/kubric:/kubric" \
  --workdir /workspace \
  kubric-gpu \
  task/task6-脚本编写/generate_s1_dataset.py --levels 1 --samples_per_level 2
```

常用参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--levels 1 2 3` | 生成哪些 level | `[1]` |
| `--samples_per_level 10` | 每个 level 生成多少样本 | 脚本内定义 |
| `--resolution 128` | 图像分辨率 | `128` |
| `--samples_per_pixel 32` | Cycles 采样数 | `32` |
| `--overwrite` | 覆盖已有数据 | 否 |
| `--views front top left` | 渲染哪些视角 | 脚本内定义 |
| `--start_id 1` | 起始样本 ID | `1` |

### S1 示例

```bash
# 生成 S1 的 level 1，100 个样本
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v "/home/lzy/project/slot-datamaking:/workspace" \
  -v "/home/lzy/project/slot-datamaking/kubric:/kubric" \
  --workdir /workspace \
  kubric-gpu \
  task/task6-脚本编写/generate_s1_dataset.py \
  --levels 1 --samples_per_level 100 --overwrite

# 生成 S1 全部 level
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v "/home/lzy/project/slot-datamaking:/workspace" \
  -v "/home/lzy/project/slot-datamaking/kubric:/kubric" \
  --workdir /workspace \
  kubric-gpu \
  task/task6-脚本编写/generate_s1_dataset.py \
  --levels 1 2 3 4 5 6 7 --overwrite
```

### S2 示例

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -v "/home/lzy/project/slot-datamaking:/workspace" \
  -v "/home/lzy/project/slot-datamaking/kubric:/kubric" \
  --workdir /workspace \
  kubric-gpu \
  task/task6-脚本编写/generate_s2_dataset.py \
  --levels 1 2 3 4 5 6 7 --overwrite
```

### S3-S8

同理，替换脚本名即可：

```bash
task/task6-脚本编写/generate_s3_dataset.py
task/task6-脚本编写/generate_s4_dataset.py
...
task/task6-脚本编写/generate_s8_dataset.py
```

## 输出目录结构

```
database/
└── S{scene}/
    └── L{level}/
        └── {sample_id}/
            ├── {id}.mp4              # 渲染视频
            ├── {id}.npz              # 深度图 (36, 128, 128) float32
            ├── video.json            # 视频元数据
            ├── object_static.json    # 物体静态属性
            └── dynamic/
                └── {frame}/          # 1-36
                    ├── {frame}.png   # 渲染帧 RGB
                    ├── force_matrix.json
                    ├── object_segment/
                    │   ├── 1.npz     # ground mask
                    │   └── 2.npz     # 动态物体 mask
                    └── object_dynamicjson/
                        ├── 1.json    # ground 物理状态
                        └── 2.json    # 动态物体物理状态
```

## 关键配置说明

### entrypoint.sh

容器入口脚本，负责：
- 设置 `PYTHONPATH` 指向挂载的 kubric 仓库
- 设置 `KUBRIC_USE_GPU=True` 启用 GPU 渲染
- 通过 `blender --background --python` 运行脚本

### blender_argv_fix.py

Blender 的 `--python` 模式会把 Blender 自身的参数混入 `sys.argv`。
此模块在脚本启动时自动清理 argv，使 argparse 正常工作。

### Dockerfile.kubric-gpu

```
nvidia/cuda:12.2.0-base-ubuntu22.04
  └─ 系统库 (libgl, libxi, libxrender, ...)
  └─ Blender 3.6.15 (/opt/blender-3.6.15-linux-x64/)
      └─ 内嵌 Python 3.10
          └─ kubric 依赖 (numpy, scipy, pybullet, trimesh, ...)
```

## GPU 渲染原理

Kubric 的 Blender 渲染器通过环境变量 `KUBRIC_USE_GPU` 控制：

```python
# kubric/renderer/blender.py
renderer.use_gpu = True  # 设置 Cycles.device = "GPU"，并选择 CUDA 计算设备
```

Blender 3.6 的 Cycles 渲染器会自动检测 CUDA 设备（RTX 4060），
使用 CUDA 12.1 内核（内嵌在 Blender 二进制中，无需额外安装 CUDA toolkit）。
默认使用 CUDA 后端；当前环境已验证 CUDA 可以用于 Cycles path tracing。

## 已知问题

### 1. 内存不足 (OOM)

Blender 渲染 + PyBullet 仿真同时运行时，内存峰值约 2-3GB。
WSL2 需要至少 12GB 内存 + 4GB swap。

配置方法：在 Windows 的 `C:\Users\{用户名}\.wslconfig`：
```ini
[wsl2]
memory=10GB
swap=8GB
```

### 2. Blender 2.93 (旧版 kubruntu) 不支持 RTX 40 系列

旧版 kubruntu 镜像中的 Blender 2.93 只支持到 RTX 30 系列（sm_86）。
必须使用本教程的 Blender 3.6 方案。

### 3. Blender 4.0+ 与 Kubric 不兼容

Blender 4.0 重命名了多个 API（材质输入名、OBJ 导入等），
kubric 代码需要大量修改。不要升级到 4.x。

### 4. OpenEXR pip 包与 Blender ABI 冲突（已解决）

**不要安装 `OpenEXR` pip 包**，会和 Blender 内置的 OpenEXR 3.0 库冲突导致 segfault。

已修补 `kubric/renderer/blender_utils.py`：
- 用 Blender 自带的 `OpenImageIO` 模块读取 multilayer EXR
- 通过通道名精确映射（Image.R, Depth.V, Vector.R 等）
- 无需任何外部 EXR 库
