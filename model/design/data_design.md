# 数据加载器设计文档

> 设计时间：2026-05-28
> 任务：T3 数据加载器

---

## 一、数据集结构

```
database/
├── S1-S8/                    # 8类物理场景
│   ├── L1-L5/               # 难度级别
│   │   ├── {sample_id}/     # 样本
│   │   │   ├── {id}.mp4     # RGB视频 (36帧, 128x128)
│   │   │   ├── {id}.npz     # 深度图 (36, 128, 128) - 暂不用
│   │   │   ├── object_static.json  # 物体静态属性
│   │   │   ├── video.json         # 视频元数据
│   │   │   └── dynamic/           # 帧级动态数据
│   │   │       └── {frame_id}/
│   │   │           ├── 1.png              # RGB帧
│   │   │           ├── force_matrix.json  # 力矩阵
│   │   │           ├── object_dynamicjson/ # 物体动态状态
│   │   │           │   └── {obj_id}.json
│   │   │           └── object_segment/    # 分割mask
│   │   │               └── {obj_id}.npz
```

## 二、各字段规格

| 字段 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| RGB视频 | [T, 3, H, W] | float32 | T=36帧, H=W=128, 归一化到[0,1] |
| 分割mask | [T, N, H, W] | float32 | N=max_objects, 二值mask |
| 物体属性 | [N, attr_dim] | float32 | 静态属性编码 |
| 动态状态 | [T, N, state_dim] | float32 | 位置/四元数/速度/角速度/合力 |
| 力矩阵 | [T, N, N, 3] | float32 | 物体间3D力向量 |

## 三、物体属性编码

### 3.1 静态属性 (object_static.json)

```python
# 数值属性直接使用
mass: float              # None for static objects → 0.0
lateralFriction: float
rollingFriction: float
spinningFriction: float
restitution: float
size: [x, y, z]         # 3D尺寸

# 类别属性编码
object_type: str         # one-hot: ground, sphere, box, cylinder
color_name: str          # one-hot or embedding

# 布尔属性
static: bool             # 0/1

# 总维度: 3(size) + 4(frictions) + 1(mass) + 1(restitution) + 1(static) + type_onehot + color
```

### 3.2 动态状态 (object_dynamicjson)

```python
position: [x, y, z]           # 3D
quaternion: [w, x, y, z]      # 4D
velocity: [vx, vy, vz]        # 3D
angular_velocity: [wx, wy, wz] # 3D
resultant_force: [fx, fy, fz]  # 3D
# 总维度: 16D
```

## 四、变长物体处理

- 每个样本物体数不同 (1~7+)
- 统一padding到 `max_objects=7`
- 输出 `valid_mask: [N]` 标记哪些是真实物体 (1=真实, 0=padding)
- ground/wall等静态物体也计入，但标记 `static=True`

## 五、时序切分

```python
history_length = 12   # 历史帧
predict_length = 12   # 预测帧
total = 24            # 从36帧中滑动窗口采样

# 滑动窗口策略
# 每个样本生成多个 (history, predict) 对
# stride=1 或 stride=6 可调
```

## 六、数据划分

```python
# 按场景分层划分
train: 70%  # 各场景均匀采样
val:   15%
test:  15%

# 按sample_id排序后划分，保证同一场景各难度都有
```

## 七、归一化策略

```python
# RGB: /255.0 → [0, 1]
# 位置/速度/力: dataset级 mean/std 归一化
# 四元数: 保持单位化，不归一化
# 尺寸: /max_size 归一化
# mask: 保持0/1
```

## 八、DataLoader输出

```python
batch = {
    'rgb': [B, T, 3, H, W],           # 完整视频
    'mask': [B, T, N, H, W],          # 分割mask
    'obj_attrs': [B, N, attr_dim],    # 静态属性
    'dyn_state': [B, T, N, state_dim], # 动态状态
    'force_matrix': [B, T, N, N, 3],  # 力矩阵
    'valid_mask': [B, N],             # 有效物体mask
    'scene_id': [B],                  # 场景编号
    'sample_id': [B],                 # 样本编号
}
```

## 九、设计决策

1. **RGB从PNG读取而非MP4**：避免视频解码依赖，直接cv2/PIL读取
2. **力矩阵从JSON读取**：需要遍历所有帧，预处理后缓存
3. **物体数padding到7**：根据数据分析，最多7个物体（含ground）
4. **滑动窗口采样**：36帧可生成多个训练对，增加数据量
5. **暂不用深度图**：用户明确说明

## 十、迭代记录

- 2026-05-28: 初始设计
- 2026-05-28: OpenCode审查超时，跳过
- 2026-05-28: Claude Code第一次审查不通过（6个问题：滑动窗口/场景分层/object_type编码/attr_dim/归一化/缓存）
- 2026-05-28: 修复所有问题：实现滑动窗口(stride=6)、场景分层划分、4种object_type one-hot、attr_dim=14、归一化策略、缓存含root_dir校验
- 2026-05-28: 14项测试全部通过
- 2026-05-28: Claude Code最终审查通过（3个轻微问题：attr_dim实际只用13列、归一化用固定值、缓存不检查文件变化，均为已知妥协）

**已知遗留问题**：
1. attr_dim声明14但实际只用13列（index 13未使用），可为color_name编码
2. 归一化使用固定值而非数据统计
3. 缓存仅校验root_dir，不检查文件变化
