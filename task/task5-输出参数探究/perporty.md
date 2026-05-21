## 视频
video.json
```json
{
  "duration_s": 3.0,
  "fps": 12,
  "num_frames": 36,
  "gravity": [0.0, 0.0, -9.8],
  "units": {
    "length": "m",
    "mass": "kg",
    "time": "s",
    "linear_velocity": "m/s",
    "angular_velocity": "rad/s",
    "angle": "rad",
    "force": "kg⋅m/s^2"
  },
  "cameras": [
    {
      "view_name": "top",
      "position": [0.0, 0.0, 5.0],
      "look_at": [0.0, 0.0, 0.0],
      "fov": 50.0,
      "resolution": [128, 128],
      "video_path": "video/top.mp4",
      "depth_path": "depth_path/top.npz"
    }
   ]
}
```


# 注意这里的物体包含地面

## 物体静态属性
object_static.json
```json
{
  "object_id": 1,
  "object_type": "sphere",
  "static": false,
  "radius": 0.22,
  "size": null,
  "height": null,
  "mass": 1.0,
  "lateralFriction": 0.0,
  "rollingFriction": 0.0,
  "spinningFriction": 0.0,
  "restitution": 1.0,
  "color_name": "red",
  "rgba": [0.86, 0.1, 0.08, 1.0]
}
```
## 物体动态属性
object_id.json
```json
{
  "object_id": 1,
  "time":3,
  "visible_area":1,
  "position": [-0.85, 0.0, 0.22],
  "quaternion": [1.0, 0.0, 0.0, 0.0],
  "velocity": [1.8, 0.0, 0.0],
  "angular_velocity": [0.0, 0.0, 0.0],
  "segmentation_path": "../object_segment/1.npz"
}
```

## 作用力矩阵_二维矩阵：最多才9个物体，10*10够用了
force_matrix.json
10*10的二维矩阵

