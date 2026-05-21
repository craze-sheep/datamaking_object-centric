## 视频
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
    "angle": "rad"
  },
  "cameras": [
    {
      "view_name": "top",
      "position": [0.0, 0.0, 5.0],
      "look_at": [0.0, 0.0, 0.0],
      "fov": 50.0,
      "resolution": [64, 64],
      "video_path": "videos/S5_L1_seed000123_top.mp4",
      "segmentation_path": "masks/S5_L1_seed000123_top.npz",
      "depth_path": "depth/S5_L1_seed000123_top.npz"
    }
   ],
}

## 地面属性
```json


```

## 物体静态属性
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
  "rollingFriction": 0.0,
  "spinningFriction": 0.0,
  "restitution": 1.0,
  "color_name": "red",
  "rgba": [0.86, 0.1, 0.08, 1.0],
}
```
## 物体动态属性
```json
{
  "object_id": "sphere_a",
  "time":3,
  "visible_area":1,
  "position": [-0.85, 0.0, 0.22],
  "quaternion": [1.0, 0.0, 0.0, 0.0],
  "velocity": [1.8, 0.0, 0.0],
  "angular_velocity": [0.0, 0.0, 0.0],
}
```
