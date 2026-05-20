# 输出JSON格式完整规范

## 1. 总体结构

```json
{
  "schema_version": "slot-datamaking-meta-v1",
  "physical_sample_id": "S5_L1_seed000123",
  "video_id": "S5_L1_seed000123_top",
  "scene_type": "S5_球撞球",
  "level_id": "L1",
  "level_name": "恢复系数的影响",
  "random_seed": 123,
  "duration_s": 3.0,
  "fps": 12,
  "num_frames": 36,
  "resolution": [64, 64],
  "gravity": [0.0, 0.0, -9.8],
  "units": { ... },
  "engine": { ... },
  "sampled_params": { ... },
  "objects": [ ... ],
  "cameras": [ ... ],
  "events": [ ... ],
  "derived_labels": { ... },
  "quality_checks": { ... },
  "state_file": "states/S5_L1_seed000123.npz"
}
```

## 2. 全局元数据字段

| 字段 | 类型 | 必须 | 说明 |
|---|---|---|---|
| `schema_version` | string | 是 | 固定值 `slot-datamaking-meta-v1` |
| `physical_sample_id` | string | 是 | 物理配置唯一ID，格式 `{scene_code}_{level_id}_seed{seed:06d}` |
| `video_id` | string | 是 | 单视角视频唯一ID，格式 `{sample_id}_{camera_view}` |
| `scene_type` | string | 是 | 场景类型：`S1_落体`、`S2_水平地面滑动`、`S3_斜面滑动`、`S4_墙面反弹`、`S5_球撞球`、`S6_球撞方块`、`S7_三物体连锁碰撞`、`S8_泛化样本` |
| `level_id` | string | 是 | 等级ID：`L1`-`L12` |
| `level_name` | string | 是 | 等级中文名 |
| `random_seed` | int | 是 | 随机种子，用于复现 |
| `duration_s` | float | 是 | 仿真时长（秒），固定3.0 |
| `fps` | int | 是 | 帧率，固定12 |
| `num_frames` | int | 是 | 总帧数，固定36 |
| `resolution` | list[int] | 是 | 视频分辨率 `[64, 64]` |
| `gravity` | list[float] | 是 | 重力向量 `[0.0, 0.0, -9.8]` |
| `units` | object | 是 | 单位系统定义 |
| `engine` | object | 是 | 物理引擎信息 |
| `sampled_params` | object | 是 | 采样参数 |
| `state_file` | string | 否 | 逐帧状态文件路径 |

## 3. 单位系统

```json
{
  "units": {
    "length": "m",
    "mass": "kg",
    "time": "s",
    "linear_velocity": "m/s",
    "angular_velocity": "rad/s",
    "angle": "rad"
  }
}
```

## 4. 物理引擎信息

```json
{
  "engine": {
    "physics": "PyBullet",
    "renderer": "Kubric/Blender",
    "docker_image": "kubricdockerhub/kubruntu",
    "notes": "Record exact versions at generation time."
  }
}
```

## 5. 采样参数

`sampled_params` 记录仿真前确定的输入配置，是生成该样本的采样参数。

```json
{
  "sampled_params": {
    "main_variable": "sphere_b 的 restitution",
    "sphere_b 的 restitution": 0.8,
    "initial_speed": 1.8,
    "impact_offset_y": 0.0
  }
}
```

**注意：** `sampled_params` 与 `derived_labels` 必须分开，不能混淆。

## 6. 物体属性（objects数组）

### 6.1 通用字段

每个物体输出成JSON对象，包含身份、几何、材质、初始状态、视觉属性：

```json
{
  "object_id": "sphere_a",
  "object_type": "sphere",
  "static": false,
  "radius": 0.22,
  "size": null,
  "height": null,
  "mass": 1.0,
  "friction": 0.0,
  "lateralFriction": 0.5,
  "rollingFriction": 0.0,
  "spinningFriction": 0.0,
  "restitution": 1.0,
  "initial_position": [-0.85, 0.0, 0.22],
  "initial_quaternion": [1.0, 0.0, 0.0, 0.0],
  "initial_velocity": [1.8, 0.0, 0.0],
  "initial_angular_velocity": [0.0, 0.0, 0.0],
  "color_name": "red",
  "rgba": [0.86, 0.1, 0.08, 1.0],
  "visible_area_by_view": {
    "top": 36,
    "front": 36
  },
  "left_visible_area_by_view": {
    "top": false,
    "front": false
  },
  "visible_area_source": "segmentation"
}
```

### 6.2 字段约定

| 字段 | 说明 |
|---|---|
| `object_id` | 视频内唯一，如 `sphere_a`、`cube_b`、`ground` |
| `object_type` | `sphere`、`cube`、`cylinder`、`ramp`、`wall` |
| `static` | `true`/`false` |
| `friction` | 动态物体表面摩擦 |
| `lateralFriction` | 静态接触面摩擦（ground/ramp/wall） |
| `initial_quaternion` | Kubric格式 `[w,x,y,z]` |
| `visible_area_source` | `segmentation`/`projected_bbox`/`not_computed` |

### 6.3 几何参数规则

| 物体类型 | 必须字段 |
|---|---|
| sphere | `radius` |
| cube/wall/ground/ramp | `size` |
| cylinder | `radius`、`height` |
| ramp | `incline_angle`、`ramp_s`、`ramp_y`、`ramp_normal` |

### 6.4 object_id 命名规范

| 场景 | object_id |
|---|---|
| 地面 | `ground` |
| 墙 | `wall`、`left_wall`、`right_wall` |
| 斜面 | `ramp` |
| 障碍物 | `obstacle_0`、`obstacle_1` |
| S1 单物体 | `object` |
| S5 球撞球 | `sphere_a`、`sphere_b` |
| S6 球撞方块 | `sphere_a`、`cube_b` |
| S7 三物体 | `object_a`、`object_b`、`object_c` |
| S8 多物体 | `object_0`、`object_1`、... |

### 6.5 物体示例

#### 地面 (ground)
```json
{
  "object_id": "ground",
  "object_type": "cube",
  "static": true,
  "size": [8.0, 6.0, 0.08],
  "initial_position": [0.0, 0.0, -0.04],
  "initial_quaternion": [1.0, 0.0, 0.0, 0.0],
  "mass": 1.0,
  "lateralFriction": 0.5,
  "rollingFriction": 0.0,
  "spinningFriction": 0.0,
  "restitution": 0.0,
  "color_name": "gray",
  "rgba": [0.55, 0.55, 0.55, 1.0]
}
```

#### 斜面 (ramp)
```json
{
  "object_id": "ramp",
  "object_type": "cube",
  "static": true,
  "size": [1.8, 0.7, 0.08],
  "incline_angle": 0.35,
  "ramp_s": -0.45,
  "ramp_y": 0.0,
  "ramp_normal": [0.34, 0.0, 0.94],
  "initial_position": [-0.35, 0.0, 0.62],
  "initial_quaternion": [0.9239, 0.0, 0.0, 0.3827],
  "mass": 1.0,
  "lateralFriction": 0.3,
  "restitution": 0.0,
  "color_name": "brown",
  "rgba": [0.6, 0.4, 0.2, 1.0]
}
```

#### 墙壁 (wall)
```json
{
  "object_id": "wall",
  "object_type": "cube",
  "static": true,
  "size": [0.08, 1.8, 0.7],
  "initial_position": [1.4, 0.0, 0.6],
  "initial_quaternion": [1.0, 0.0, 0.0, 0.0],
  "mass": 1.0,
  "lateralFriction": 0.5,
  "restitution": 0.8,
  "color_name": "white",
  "rgba": [1.0, 1.0, 1.0, 1.0]
}
```

## 7. 相机配置（cameras数组）

```json
{
  "cameras": [
    {
      "view_name": "top",
      "position": [0.0, -0.01, 8.0],
      "look_at": [0.0, 0.0, 0.0],
      "quaternion": [1.0, 0.0, 0.0, 0.0],
      "orthographic_scale": 5.0,
      "fov": null,
      "focal_length": null,
      "resolution": [64, 64],
      "video_path": "videos/top.mp4",
      "segmentation_path": "masks/top_segmentation.npz",
      "depth_path": "depth/top_depth.npz"
    },
    {
      "view_name": "front",
      "position": [0.0, -7.5, 3.2],
      "look_at": [0.0, 0.0, 0.35],
      "quaternion": [0.966, 0.0, 0.259, 0.0],
      "fov": 50.0,
      "focal_length": 35,
      "sensor_width": 32,
      "resolution": [64, 64],
      "video_path": "videos/front.mp4",
      "segmentation_path": "masks/front_segmentation.npz",
      "depth_path": "depth/front_depth.npz"
    }
  ]
}
```

### 视角配置

| 视角 | 相机类型 | 位置 | 朝向 | 参数 |
|---|---|---|---|---|
| `front` | Perspective | (0, -7.5, 3.2) | (0, 0, 0.35) | focal_length=35, sensor_width=32 |
| `back` | Perspective | (0, 7.5, 3.2) | (0, 0, 0.35) | focal_length=35, sensor_width=32 |
| `left` | Perspective | (-7.5, 0, 3.2) | (0, 0, 0.35) | focal_length=35, sensor_width=32 |
| `right` | Perspective | (7.5, 0, 3.2) | (0, 0, 0.35) | focal_length=35, sensor_width=32 |
| `top` | Orthographic | (0, -0.01, 8.0) | (0, 0, 0) | orthographic_scale=5.0 |

### 场景视角子集

| 场景 | 视角 |
|---|---|
| S1/S4 | front, top (2个) |
| S2/S3 | front, top, left (3个) |
| S5/S6/S7/S8 | front, back, left, right, top (5个) |

## 8. 事件记录（events数组）

### 8.1 统一事件结构

所有事件都用同一个基本结构：

```json
{
  "event_id": "evt_0001",
  "event_type": "dynamic_dynamic_contact",
  "time": 0.5833,
  "frame": 7,
  "substep": 12,
  "objects": ["sphere_a", "sphere_b"],
  "position": [0.33, 0.0, 0.22],
  "normal": [1.0, 0.0, 0.0],
  "metrics": {
    "relative_speed_before": 1.8,
    "relative_speed_after": 1.44,
    "normal_speed_before": 1.8,
    "normal_speed_after": 1.44,
    "tangential_speed_before": 0.0,
    "tangential_speed_after": 0.0
  },
  "pre_state": {
    "sphere_a": {
      "position": [-0.01, 0.0, 0.22],
      "linear_velocity": [1.8, 0.0, 0.0],
      "angular_velocity": [0.0, 0.0, 0.0]
    },
    "sphere_b": {
      "position": [0.43, 0.0, 0.22],
      "linear_velocity": [0.0, 0.0, 0.0],
      "angular_velocity": [0.0, 0.0, 0.0]
    }
  },
  "post_state": {
    "sphere_a": {
      "linear_velocity": [0.18, 0.0, 0.0],
      "angular_velocity": [0.0, 0.0, 0.0]
    },
    "sphere_b": {
      "linear_velocity": [1.62, 0.0, 0.0],
      "angular_velocity": [0.0, 0.0, 0.0]
    }
  }
}
```

### 8.2 事件必须有的字段

| 字段 | 必须 | 说明 |
|---|---|---|
| `event_id` | 是 | `evt_0001` |
| `event_type` | 是 | 统一枚举 |
| `time` | 是 | 秒 |
| `frame` | 是 | 12fps帧号 |
| `objects` | 是 | 参与对象 |
| `position` | 接触事件必须 | 接触点或事件点 |
| `normal` | 接触事件必须 | 接触法线 |
| `metrics` | 建议 | 事件相关指标 |
| `pre_state` | 接触事件必须 | 碰前状态 |
| `post_state` | 接触事件必须 | 碰后状态 |

### 8.3 事件类型枚举

| 类别 | event_type | 说明 |
|---|---|---|
| 接触 | `first_ground_contact` | 首次地面接触 |
| 接触 | `first_wall_contact` | 首次墙面接触 |
| 接触 | `dynamic_dynamic_contact` | 动态物体间接触 |
| 接触 | `dynamic_static_contact` | 动态与静态物体接触 |
| 运动状态 | `slide_start` | 开始滑动 |
| 运动状态 | `rolling_transition` | 滚动转换 |
| 运动状态 | `stop` | 停止运动 |
| 运动状态 | `rebound_peak` | 反弹峰值 |
| 运动状态 | `ramp_bottom_reached` | 到达斜面底部 |
| 运动状态 | `leave_visible_area` | 离开可视区域 |
| 轨迹关系 | `closest_approach` | 最接近点 |
| 轨迹关系 | `shared_region_crossing` | 共享区域交叉 |
| 轨迹关系 | `path_intersection_without_contact` | 路径交叉无接触 |
| 干预 | `external_impulse` | 外部冲量 |
| 过滤 | `invalid_initial_overlap` | 初始重叠无效 |
| 过滤 | `missed_required_contact` | 缺失必需接触 |
| 过滤 | `unexpected_dynamic_contact` | 意外动态接触 |

### 8.4 S7 contact_pair_sequence

S7必须记录完整接触序列：

```json
{
  "contact_pair_sequence": [
    {"pair": ["object_a", "object_b"], "time": 0.42, "frame": 5},
    {"pair": ["object_b", "object_c"], "time": 0.75, "frame": 9},
    {"pair": ["object_c", "right_wall"], "time": 1.22, "frame": 15}
  ]
}
```

## 9. 派生标签（derived_labels）

`derived_labels` 由仿真结果或后处理得到，不能只从采样参数猜。

### 9.1 通用标签（所有场景）

```json
{
  "derived_labels": {
    "final_positions": {
      "sphere_a": [0.42, 0.0, 0.22],
      "sphere_b": [3.0, 0.0, 0.22]
    },
    "final_velocities": {
      "sphere_a": [0.18, 0.0, 0.0],
      "sphere_b": [1.62, 0.0, 0.0]
    },
    "final_linear_speeds": {
      "sphere_a": 0.18,
      "sphere_b": 1.62
    },
    "final_angular_speeds": {
      "sphere_a": 0.0,
      "sphere_b": 0.0
    },
    "is_static_at_end": {
      "sphere_a": false,
      "sphere_b": false
    },
    "object_left_visible_area": {
      "sphere_a": false,
      "sphere_b": true
    },
    "contact_counts": {
      "dynamic_dynamic_new": 1,
      "dynamic_static_new": 2,
      "dynamic_static_resting_frames": 72,
      "total_new": 3
    },
    "valid_sample": true,
    "invalid_reason": null,
    "stress_sample": false,
    "sample_split": "main"
  }
}
```

### 9.2 碰撞特有标签（S4/S5/S6/S7）

```json
{
  "derived_labels": {
    "first_contact_time": 0.5833,
    "first_contact_frame": 7,
    "pre_collision_velocity_a": [1.8, 0.0, 0.0],
    "pre_collision_velocity_b": [0.0, 0.0, 0.0],
    "post_collision_velocity_a": [0.18, 0.0, 0.0],
    "post_collision_velocity_b": [1.62, 0.0, 0.0],
    "incident_speed_normal": 1.8,
    "separation_speed_normal": 1.44,
    "effective_restitution": 0.8,
    "effective_restitution_source": "substep_contact_state",
    "momentum_before": [1.8, 0.0, 0.0],
    "momentum_after": [1.8, 0.0, 0.0],
    "kinetic_energy_before": 1.62,
    "kinetic_energy_after": 1.3284,
    "scattering_angle_a": 0.0,
    "scattering_angle_b": 0.0
  }
}
```

### 9.3 S1特有标签

```json
{
  "derived_labels": {
    "first_contact_time": 0.42,
    "first_contact_frame": 5,
    "effective_restitution": 0.5,
    "landing_position": [0.0, 0.0, 0.0],
    "rebound_height": 0.35,
    "max_height_after_release": 1.0,
    "release_clearance": 0.78
  }
}
```

### 9.4 S2特有标签

```json
{
  "derived_labels": {
    "stop_time": 2.17,
    "stop_frame": 26,
    "travel_distance": 1.85,
    "displacement_vector": [1.85, 0.0, 0.0],
    "effective_lateral_friction": 0.4,
    "rolling_transition_time": 0.83
  }
}
```

### 9.5 S8特有标签

```json
{
  "derived_labels": {
    "object_count": 3,
    "dynamic_object_types": ["sphere", "cube", "sphere"],
    "no_dynamic_collision": true,
    "total_dynamic_dynamic_contact_count": 0,
    "total_dynamic_static_contact_count": 2,
    "min_pairwise_distance": 0.51,
    "min_pairwise_distance_frame": 16,
    "closest_pair": ["object_0", "object_2"],
    "closest_approach_time": 1.333,
    "closest_approach_frame": 16,
    "relative_velocity_at_closest_approach": [0.8, 0.0, 0.0],
    "negative_reason": "spatial_separation",
    "counterfactual_would_collide": false
  }
}
```

### 9.6 S8 negative_reason 枚举

```
spatial_separation
temporal_separation
relative_speed_separation
random_no_collision
blocked_by_static_obstacle
stopped_by_friction
height_separation
moving_shield
boundary_phase_miss
near_miss_margin
orientation_miss
external_impulse_avoidance
static_scene
diverging_velocities
```

### 9.7 effective_restitution 计算

定义：碰后法向分离速度 / 碰前法向接近速度

```python
def effective_restitution(v_a_pre, v_b_pre, v_a_post, v_b_post, normal):
    rel_pre = dot(v_a_pre - v_b_pre, normal)
    rel_post = dot(v_b_post - v_a_post, normal)
    if abs(rel_pre) < 1e-6:
        return None
    return max(0.0, rel_post / abs(rel_pre))
```

**注意：**
- 墙面碰撞时，墙速度为0
- 地面反弹时，法线通常是 `[0,0,1]`
- 应优先使用引擎substep级接触前后速度计算
- 如果没有substep state，标记为 `effective_restitution_source="frame_approx"`

## 10. 质量检查（quality_checks）

### 10.1 完整结构

```json
{
  "quality_checks": {
    "valid_sample": true,
    "invalid_reason": null,
    "stress_sample": false,
    "sample_split": "main",
    "initial_overlap": false,
    "initial_ground_penetration": false,
    "required_contact_sequence_satisfied": true,
    "unexpected_dynamic_contact": false,
    "required_no_dynamic_collision": false,
    "contact_counts": {
      "dynamic_dynamic_new": 1,
      "dynamic_static_new": 2,
      "dynamic_static_resting_frames": 72,
      "total_new": 3
    },
    "min_visible_frames_by_object": {
      "sphere_a": 36,
      "sphere_b": 24
    },
    "warnings": [
      "sphere_b leaves visible area after enough post-contact frames; keep as boundary sample."
    ],
    "thresholds": {
      "static_linear_speed_threshold": 0.03,
      "static_angular_speed_threshold": 0.05,
      "contact_time_tolerance": 0.0833,
      "visible_area_min_frames": 12
    }
  }
}
```

### 10.2 valid_sample 判定规则

按场景：

```python
if scene_type in ["S5_球撞球", "S6_球撞方块"]:
    valid = first_required_contact is not None

if scene_type == "S7_三物体连锁碰撞":
    valid = required_contact_sequence_satisfied

if scene_type == "S8_泛化样本":
    valid = total_dynamic_dynamic_contact_count == 0

if scene_type in ["S1_落体", "S2_水平地面滑动", "S3_斜面滑动", "S4_墙面反弹"]:
    valid = not initial_overlap and not render_failed
```

### 10.3 invalid_reason 枚举

```
initial_overlap
initial_ground_penetration
missed_required_contact
wrong_contact_order
unexpected_dynamic_contact
object_left_visible_area_too_early
render_failed
physics_exploded
nan_state
too_small_visible_area
```

## 11. 逐帧状态文件（states.npz）

### 11.1 数组结构

```python
{
  "object_ids": np.array,              # [num_objects] object_id列表
  "is_static": np.array,               # [num_objects] 是否静态
  "positions": np.array,               # [num_frames, num_objects, 3]
  "quaternions_wxyz": np.array,        # [num_frames, num_objects, 4]
  "linear_velocities": np.array,       # [num_frames, num_objects, 3]
  "angular_velocities": np.array,      # [num_frames, num_objects, 3]
  "visible_area": np.array,            # [num_frames, num_views, num_objects]
  "in_view": np.array,                 # [num_frames, num_views, num_objects]
  "contact_matrix": np.array,          # [num_frames, num_objects, num_objects]
  "dynamic_static_contact": np.array   # [num_frames, num_objects]
}
```

### 11.2 为什么逐帧状态必要

逐帧状态用于计算：
- `first_contact_time`
- `stop_time`
- `travel_distance`
- `rebound_height`
- `closest_approach_time`
- `scattering_angle`
- `visible_area_by_view`
- `left_visible_area`

如果只保存最后标签，后续发现标签算法错了，就无法重算。

## 12. 文件组织

### 12.1 正式数据集输出目录结构

```
dataset_output/
  manifest.json
  metadata.jsonl
  splits/
    train.json
    val.json
    test.json
    debug.json
  samples/
    {scene_type}/
      {level_id}/
        {sample_id}/
          metadata.json
          states.npz
          videos/
            {view}.mp4
          masks/                 # 可选
            {view}_segmentation.npz
          depth/                 # 可选
            {view}_depth.npz
          thumbnails/            # 可选
            {view}.png
          logs/
            warnings.json
  invalid_samples/
    {scene_type}/
      {level_id}/
        {sample_id}/
          metadata.json
          states.npz
          logs/
  _tmp/
  _cache/
  _logs/
```

### 12.2 路径记录规范

metadata中所有路径使用相对 `dataset_output/` 的相对路径：

```json
{
  "video_path": "samples/S5_球撞球/L1/S5_L1_seed000123/videos/top.mp4",
  "metadata_path": "samples/S5_球撞球/L1/S5_L1_seed000123/metadata.json",
  "states_path": "samples/S5_球撞球/L1/S5_L1_seed000123/states.npz"
}
```

不建议写绝对路径，原因：
- 换机器后路径失效
- 打包发布不方便
- Docker内外路径不同

### 12.3 global metadata.jsonl

每行一条单视角视频记录：

```json
{
  "video_id": "S5_L1_seed000123_top",
  "physical_sample_id": "S5_L1_seed000123",
  "scene_type": "S5_球撞球",
  "level_id": "L1",
  "camera_view": "top",
  "video_path": "samples/S5_球撞球/L1/S5_L1_seed000123/videos/top.mp4",
  "metadata_path": "samples/S5_球撞球/L1/S5_L1_seed000123/metadata.json",
  "states_path": "samples/S5_球撞球/L1/S5_L1_seed000123/states.npz",
  "valid_sample": true,
  "stress_sample": false,
  "sample_split": "main",
  "first_contact_frame": 7,
  "object_count": 2
}
```

## 13. 完整示例（S5 球撞球）

```json
{
  "schema_version": "slot-datamaking-meta-v1",
  "physical_sample_id": "S5_L1_seed000123",
  "video_id": "S5_L1_seed000123_top",
  "scene_type": "S5_球撞球",
  "level_id": "L1",
  "level_name": "恢复系数的影响",
  "random_seed": 123,
  "duration_s": 3.0,
  "fps": 12,
  "num_frames": 36,
  "resolution": [64, 64],
  "gravity": [0.0, 0.0, -9.8],
  "units": {
    "length": "m",
    "mass": "kg",
    "time": "s",
    "linear_velocity": "m/s",
    "angular_velocity": "rad/s",
    "angle": "rad"
  },
  "engine": {
    "physics": "PyBullet",
    "renderer": "Kubric/Blender",
    "docker_image": "kubricdockerhub/kubruntu"
  },
  "sampled_params": {
    "main_variable": "sphere_b 的 restitution",
    "sphere_b 的 restitution": 0.8,
    "initial_speed": 1.8,
    "impact_offset_y": 0.0
  },
  "objects": [
    {
      "object_id": "ground",
      "object_type": "cube",
      "static": true,
      "size": [8.0, 6.0, 0.08],
      "initial_position": [0.0, 0.0, -0.04],
      "initial_quaternion": [1.0, 0.0, 0.0, 0.0],
      "mass": 1.0,
      "lateralFriction": 0.5,
      "rollingFriction": 0.0,
      "spinningFriction": 0.0,
      "restitution": 0.0,
      "color_name": "gray",
      "rgba": [0.55, 0.55, 0.55, 1.0]
    },
    {
      "object_id": "sphere_a",
      "object_type": "sphere",
      "static": false,
      "radius": 0.22,
      "initial_position": [-0.85, 0.0, 0.22],
      "initial_quaternion": [1.0, 0.0, 0.0, 0.0],
      "initial_velocity": [1.8, 0.0, 0.0],
      "initial_angular_velocity": [0.0, 0.0, 0.0],
      "mass": 1.0,
      "friction": 0.0,
      "rollingFriction": 0.0,
      "spinningFriction": 0.0,
      "restitution": 1.0,
      "color_name": "red",
      "rgba": [0.86, 0.1, 0.08, 1.0],
      "visible_area_by_view": {"top": 36, "front": 36},
      "left_visible_area_by_view": {"top": false, "front": false},
      "visible_area_source": "segmentation"
    },
    {
      "object_id": "sphere_b",
      "object_type": "sphere",
      "static": false,
      "radius": 0.22,
      "initial_position": [0.55, 0.0, 0.22],
      "initial_quaternion": [1.0, 0.0, 0.0, 0.0],
      "initial_velocity": [0.0, 0.0, 0.0],
      "initial_angular_velocity": [0.0, 0.0, 0.0],
      "mass": 1.0,
      "friction": 0.0,
      "rollingFriction": 0.0,
      "spinningFriction": 0.0,
      "restitution": 0.8,
      "color_name": "blue",
      "rgba": [0.08, 0.2, 0.86, 1.0],
      "visible_area_by_view": {"top": 24, "front": 24},
      "left_visible_area_by_view": {"top": true, "front": true},
      "visible_area_source": "segmentation"
    }
  ],
  "cameras": [
    {
      "view_name": "top",
      "position": [0.0, -0.01, 8.0],
      "look_at": [0.0, 0.0, 0.0],
      "orthographic_scale": 5.0,
      "resolution": [64, 64],
      "video_path": "videos/top.mp4",
      "segmentation_path": "masks/top_segmentation.npz",
      "depth_path": "depth/top_depth.npz"
    },
    {
      "view_name": "front",
      "position": [0.0, -7.5, 3.2],
      "look_at": [0.0, 0.0, 0.35],
      "focal_length": 35,
      "sensor_width": 32,
      "resolution": [64, 64],
      "video_path": "videos/front.mp4",
      "segmentation_path": "masks/front_segmentation.npz",
      "depth_path": "depth/front_depth.npz"
    }
  ],
  "events": [
    {
      "event_id": "evt_0001",
      "event_type": "dynamic_dynamic_contact",
      "time": 0.5833,
      "frame": 7,
      "objects": ["sphere_a", "sphere_b"],
      "position": [0.33, 0.0, 0.22],
      "normal": [1.0, 0.0, 0.0],
      "metrics": {
        "relative_speed_before": 1.8,
        "relative_speed_after": 1.44,
        "normal_speed_before": 1.8,
        "normal_speed_after": 1.44
      },
      "pre_state": {
        "sphere_a": {
          "position": [-0.01, 0.0, 0.22],
          "linear_velocity": [1.8, 0.0, 0.0],
          "angular_velocity": [0.0, 0.0, 0.0]
        },
        "sphere_b": {
          "position": [0.43, 0.0, 0.22],
          "linear_velocity": [0.0, 0.0, 0.0],
          "angular_velocity": [0.0, 0.0, 0.0]
        }
      },
      "post_state": {
        "sphere_a": {
          "linear_velocity": [0.18, 0.0, 0.0],
          "angular_velocity": [0.0, 0.0, 0.0]
        },
        "sphere_b": {
          "linear_velocity": [1.62, 0.0, 0.0],
          "angular_velocity": [0.0, 0.0, 0.0]
        }
      }
    }
  ],
  "derived_labels": {
    "first_contact_time": 0.5833,
    "first_contact_frame": 7,
    "pre_collision_velocity_a": [1.8, 0.0, 0.0],
    "pre_collision_velocity_b": [0.0, 0.0, 0.0],
    "post_collision_velocity_a": [0.18, 0.0, 0.0],
    "post_collision_velocity_b": [1.62, 0.0, 0.0],
    "incident_speed_normal": 1.8,
    "separation_speed_normal": 1.44,
    "effective_restitution": 0.8,
    "effective_restitution_source": "substep_contact_state",
    "momentum_before": [1.8, 0.0, 0.0],
    "momentum_after": [1.8, 0.0, 0.0],
    "kinetic_energy_before": 1.62,
    "kinetic_energy_after": 1.3284,
    "scattering_angle_a": 0.0,
    "scattering_angle_b": 0.0,
    "final_positions": {
      "sphere_a": [0.42, 0.0, 0.22],
      "sphere_b": [3.0, 0.0, 0.22]
    },
    "final_velocities": {
      "sphere_a": [0.18, 0.0, 0.0],
      "sphere_b": [1.62, 0.0, 0.0]
    },
    "final_linear_speeds": {
      "sphere_a": 0.18,
      "sphere_b": 1.62
    },
    "final_angular_speeds": {
      "sphere_a": 0.0,
      "sphere_b": 0.0
    },
    "is_static_at_end": {
      "sphere_a": false,
      "sphere_b": false
    },
    "object_left_visible_area": {
      "sphere_a": false,
      "sphere_b": true
    },
    "contact_counts": {
      "dynamic_dynamic_new": 1,
      "dynamic_static_new": 2,
      "dynamic_static_resting_frames": 72,
      "total_new": 3
    },
    "valid_sample": true,
    "invalid_reason": null,
    "stress_sample": false,
    "sample_split": "main"
  },
  "quality_checks": {
    "valid_sample": true,
    "invalid_reason": null,
    "stress_sample": false,
    "sample_split": "main",
    "initial_overlap": false,
    "required_contact_sequence_satisfied": true,
    "unexpected_dynamic_contact": false,
    "contact_counts": {
      "dynamic_dynamic_new": 1,
      "dynamic_static_new": 2,
      "dynamic_static_resting_frames": 72,
      "total_new": 3
    },
    "min_visible_frames_by_object": {
      "sphere_a": 36,
      "sphere_b": 24
    },
    "warnings": [
      "sphere_b leaves visible area after enough post-contact frames; keep as boundary sample."
    ],
    "thresholds": {
      "static_linear_speed_threshold": 0.03,
      "static_angular_speed_threshold": 0.05,
      "contact_time_tolerance": 0.0833,
      "visible_area_min_frames": 12
    }
  },
  "state_file": "states/S5_L1_seed000123.npz"
}
```

## 14. 强制约定

1. 所有时间字段同时写 `time` 和 `frame`
2. 所有速度字段保留向量，不只写 speed
3. 所有 contact 事件写对象 pair、normal、position
4. S7 必须写完整 `contact_pair_sequence`
5. S8 必须写 `negative_reason` 和 `total_dynamic_dynamic_contact_count`
6. 出界不是错误，但必须写 `left_visible_area` 和可见帧数
7. `sampled_params` 和 `derived_labels` 必须分开
8. `valid_sample=false` 的 metadata 也建议保留到 invalid 目录
9. metadata 里路径一律相对数据集根目录
10. quaternion 对外统一 `[w,x,y,z]`

## 15. 记录责任划分

| 阶段 | 负责记录 |
|---|---|
| 采样器 | `sampled_params`、`objects` 初始属性、相机、seed |
| 物理仿真 | `states`、contact events、final states |
| 后处理 | first_contact、stop、closest_approach、visible-area、valid_sample |
| 渲染器 | video_path、depth、segmentation、camera matrices |
| 质检脚本 | invalid_reason、warning、人工抽检结果 |
