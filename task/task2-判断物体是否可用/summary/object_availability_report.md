# task2 判断物体是否可用

## 结论摘要

- 总计检查对象/对象集合：35
- OK：11，当前环境可直接加载或已有本地资源。
- OK*：23，对象定义或远端 asset id 可用，但当前 Kubric 环境需补依赖/下载资产。
- CHECK：1，当前工作区未完成验证，不建议直接纳入第一批生成。

当前 `python3` 导入 Kubric 失败，错误为缺少 `pyquaternion`；因此基础 Kubric primitive 和 KuBasic 标为 OK*，不是物体不可用，而是渲染环境还没完整跑通。

## 建议

- 第一批可以先用 `sphere/cube/cylinder/wall/ramp/occluder/pillar` 和 pybullet_data 中已加载成功的 URDF。
- KuBasic 的 `cone/torus/gear/torus_knot/sponge/spot/teapot/suzanne` 可以作为第二批，先补 Kubric 依赖并缓存资产。
- GSO 暂时放到第三批；本次没有本地 manifest，远端 manifest 请求失败，需要单独下载和抽样验证。
- `random_urdfs/000-999` 本地存在 1000 个，但建议抽样剔除尺寸异常、重心怪、碰撞不稳定的物体。

## 分类目录

| group | 中文名称 | 数量 | 目录 |
|---|---|---:|---|
| `basic_dynamic` | 基础动态物体 | 9 | `../by_category/basic_dynamic/` |
| `basic_static` | 基础静态物体 | 5 | `../by_category/basic_static/` |
| `kubasic` | KuBasic 复杂几何体 | 8 | `../by_category/kubasic/` |
| `pybullet_urdf` | PyBullet 常用 URDF 物体 | 10 | `../by_category/pybullet_urdf/` |
| `pybullet_random_urdfs` | PyBullet 随机 URDF 物体库 | 1 | `../by_category/pybullet_random_urdfs/` |
| `gso` | Google Scanned Objects | 1 | `../by_category/gso/` |
| `scenario_template` | 场景模板 | 1 | `../by_category/scenario_template/` |

## 明细

| object_id | group | scenes | status | 结论 | 证据 |
|---|---|---|---|---|---|
| `sphere_s` | basic_dynamic | S1/S4 | OK* | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 | 基础几何体可由 PyBullet 原生碰撞体创建，Kubric 源码也提供 Sphere/Cube/Cylinder。 |
| `sphere_m` | basic_dynamic | S1/S3/S5-S8/S13/S14 | OK* | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 | 基础几何体可由 PyBullet 原生碰撞体创建，Kubric 源码也提供 Sphere/Cube/Cylinder。 |
| `sphere_l` | basic_dynamic | S1/S4 | OK* | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 | 基础几何体可由 PyBullet 原生碰撞体创建，Kubric 源码也提供 Sphere/Cube/Cylinder。 |
| `cube_s` | basic_dynamic | S1/S2/S14 | OK* | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 | 基础几何体可由 PyBullet 原生碰撞体创建，Kubric 源码也提供 Sphere/Cube/Cylinder。 |
| `cube_m` | basic_dynamic | S1-S3/S6/S8/S13/S14 | OK* | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 | 基础几何体可由 PyBullet 原生碰撞体创建，Kubric 源码也提供 Sphere/Cube/Cylinder。 |
| `cube_l` | basic_dynamic | S1/S2/S14 | OK* | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 | 基础几何体可由 PyBullet 原生碰撞体创建，Kubric 源码也提供 Sphere/Cube/Cylinder。 |
| `cylinder_s` | basic_dynamic | S1/S2 | OK* | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 | 基础几何体可由 PyBullet 原生碰撞体创建，Kubric 源码也提供 Sphere/Cube/Cylinder。 |
| `cylinder_m` | basic_dynamic | S1-S3/S9 | OK* | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 | 基础几何体可由 PyBullet 原生碰撞体创建，Kubric 源码也提供 Sphere/Cube/Cylinder。 |
| `cylinder_l` | basic_dynamic | S1/S2 | OK* | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 | 基础几何体可由 PyBullet 原生碰撞体创建，Kubric 源码也提供 Sphere/Cube/Cylinder。 |
| `wall_x` | basic_static | S4/S13 | OK* | 可用；属于基础静态碰撞几何，需在 Kubric 环境补齐依赖后批量渲染。 | 可用 Cube/Cylinder 参数化生成，适合墙、斜面、遮挡板和柱体。 |
| `wall_y` | basic_static | optional | OK* | 可用；属于基础静态碰撞几何，需在 Kubric 环境补齐依赖后批量渲染。 | 可用 Cube/Cylinder 参数化生成，适合墙、斜面、遮挡板和柱体。 |
| `ramp` | basic_static | S3 | OK* | 可用；属于基础静态碰撞几何，需在 Kubric 环境补齐依赖后批量渲染。 | 可用 Cube/Cylinder 参数化生成，适合墙、斜面、遮挡板和柱体。 |
| `occluder` | basic_static | S13 | OK* | 可用；属于基础静态碰撞几何，需在 Kubric 环境补齐依赖后批量渲染。 | 可用 Cube/Cylinder 参数化生成，适合墙、斜面、遮挡板和柱体。 |
| `pillar` | basic_static | S13 | OK* | 可用；属于基础静态碰撞几何，需在 Kubric 环境补齐依赖后批量渲染。 | 可用 Cube/Cylinder 参数化生成，适合墙、斜面、遮挡板和柱体。 |
| `cone` | kubasic | S9 | OK* | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 | 已读取 KuBasic public manifest，8 个目标 id 均存在。 |
| `torus` | kubasic | S9 | OK* | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 | 已读取 KuBasic public manifest，8 个目标 id 均存在。 |
| `gear` | kubasic | S9 | OK* | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 | 已读取 KuBasic public manifest，8 个目标 id 均存在。 |
| `torus_knot` | kubasic | S9 | OK* | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 | 已读取 KuBasic public manifest，8 个目标 id 均存在。 |
| `sponge` | kubasic | S9 | OK* | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 | 已读取 KuBasic public manifest，8 个目标 id 均存在。 |
| `spot` | kubasic | S9 | OK* | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 | 已读取 KuBasic public manifest，8 个目标 id 均存在。 |
| `teapot` | kubasic | S9 | OK* | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 | 已读取 KuBasic public manifest，8 个目标 id 均存在。 |
| `suzanne` | kubasic | S9 | OK* | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 | 已读取 KuBasic public manifest，8 个目标 id 均存在。 |
| `soccerball.urdf` | pybullet_urdf | S10 | OK | 当前环境可直接加载。 | pybullet_data 文件存在，并已用 p.loadURDF(DIRECT) 加载成功。 |
| `cube.urdf` | pybullet_urdf | S10 | OK | 当前环境可直接加载。 | pybullet_data 文件存在，并已用 p.loadURDF(DIRECT) 加载成功。 |
| `block.urdf` | pybullet_urdf | S10 | OK | 当前环境可直接加载。 | pybullet_data 文件存在，并已用 p.loadURDF(DIRECT) 加载成功。 |
| `lego/lego.urdf` | pybullet_urdf | S10 | OK | 当前环境可直接加载。 | pybullet_data 文件存在，并已用 p.loadURDF(DIRECT) 加载成功。 |
| `duck_vhacd.urdf` | pybullet_urdf | S10 | OK | 当前环境可直接加载。 | pybullet_data 文件存在，并已用 p.loadURDF(DIRECT) 加载成功。 |
| `teddy_vhacd.urdf` | pybullet_urdf | S10 | OK | 当前环境可直接加载。 | pybullet_data 文件存在，并已用 p.loadURDF(DIRECT) 加载成功。 |
| `objects/mug.urdf` | pybullet_urdf | S10 | OK | 当前环境可直接加载。 | pybullet_data 文件存在，并已用 p.loadURDF(DIRECT) 加载成功。 |
| `tray/tray.urdf` | pybullet_urdf | S10 | OK | 当前环境可直接加载。 | pybullet_data 文件存在，并已用 p.loadURDF(DIRECT) 加载成功。 |
| `domino/domino.urdf` | pybullet_urdf | S11 | OK | 当前环境可直接加载。 | pybullet_data 文件存在，并已用 p.loadURDF(DIRECT) 加载成功。 |
| `jenga/jenga.urdf` | pybullet_urdf | S12 | OK | 当前环境可直接加载。 | pybullet_data 文件存在，并已用 p.loadURDF(DIRECT) 加载成功。 |
| `random_urdfs/000-999` | pybullet_random_urdfs | S16 | OK | 当前环境可用；本机检测到 1000 个随机 URDF。 | 适合做杂物和静态/动态干扰物，但应抽样剔除极端尺寸或不稳定物体。 |
| `Google Scanned Objects` | gso | S15/S16 | CHECK | 当前工作区没有本地 GSO manifest；远端 manifest 本次请求失败，暂不算已验证可用。 | MOVi 脚本引用 GSO，但生成前需要下载/缓存资产并做尺寸、碰撞稳定性抽检。 |
| `bouncing_balls: sphere/cube/mixed` | scenario_template | S14 | OK* | 场景模板和物体类型设计可用；仍受当前 Kubric 依赖缺失影响。 | 对象由 sphere/cube primitive 组成，逻辑上不依赖额外资产。 |
