# Type1 参数验证报告

- 参数设置测试：112/112 成功
- smoke case：4/4 成功
- 输出目录：`outputs`

## 结论
- PyBullet 可配置地面参数：lateralFriction, rollingFriction, spinningFriction, restitution
- PyBullet 可配置物体参数：radius/size, initial_position, initial_velocity, initial_angular_velocity, mass, restitution, friction, color
- color 通过 visual shape 设置，不属于动力学参数
- 其他属性使用 PyBullet 默认值

## Smoke Cases
- level1_sphere：OK，video=`outputs/level1_sphere.mp4`，image=`outputs/level1_sphere.png`，trajectory=`outputs/level1_sphere.json`
- level2_sphere：OK，video=`outputs/level2_sphere.mp4`，image=`outputs/level2_sphere.png`，trajectory=`outputs/level2_sphere.json`
- level3_sphere：OK，video=`outputs/level3_sphere.mp4`，image=`outputs/level3_sphere.png`，trajectory=`outputs/level3_sphere.json`
- level3_cube：OK，video=`outputs/level3_cube.mp4`，image=`outputs/level3_cube.png`，trajectory=`outputs/level3_cube.json`
