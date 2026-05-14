# object简介.md 物体数量补充

来源文件：`/home/lzy/project/slot-datamaking/kubric-main/场景/object简介.md`

## 结论

如果按当前本机 `pybullet_data` 实际扫描的资源文件算：

| 口径 | 数量 | 说明 |
|---|---:|---|
| 可加载模型文件 | 1,136 | `URDF 1095 + SDF 15 + MJCF XML 26` |
| mesh 资源文件 | 1,215 | `OBJ 1117 + STL 86 + DAE 12`，通常是外观/碰撞资源，不一定每个都等价于一个独立可直接用物体 |
| 全部资源文件 | 2,351 | 上面两类相加 |
| `random_urdfs` 随机物体 | 1,000 | 已包含在 1,095 个 URDF 里 |
| 非 `random_urdfs` URDF | 95 | 常规命名 URDF，包括球、方块、机器人、桌子、杯子等 |

如果按 `object简介.md` 正文中显式点名、分组介绍的对象入口算：

| 类别 | 数量 | 是否建议放入第一版数据集 |
|---|---:|---|
| PyBullet 原生几何类型 | 7 | 推荐，最稳定 |
| 基础/场地/日常 URDF 或 SDF | 37 | 部分推荐，优先球、方块、Jenga、多米诺、杯子、托盘等 |
| 机器人和可动机构 | 40 | 当前任务不推荐，会引入主动控制外力 |
| MJCF 强化学习模型条目 | 17 | 当前任务不推荐，偏控制任务 |
| 软体/地形资源入口 | 4 | 第一版不推荐，标注和稳定性更复杂 |
| `random_urdfs/000-999` | 1,000 | 可作为后期杂物/OOD，但需要抽样筛选 |
| Kubric 兼容实现入口 | 3 | 是实现方式，不是额外资产 |

因此：

- 严格按本机资源文件：`object简介.md` 对应的 pybullet_data 资源约为 **2,351 个文件**。
- 严格按可直接加载的模型文件：约 **1,136 个模型文件**。
- 按正文点名的对象入口，并把 `random_urdfs` 展开：约 **1,105 个 PyBullet 对象/资源入口**；如果连 Kubric 兼容入口也算上，则是 **1,108 个入口**。
- 按正文点名但不展开 `random_urdfs`：约 **106 个 PyBullet 对象/资源入口**；加 Kubric 兼容入口为 **109 个入口**。

> 注意：`object简介.md` 旧文中写了 `STL mesh = 71`，我重新扫描当前环境得到的是 `STL mesh = 86`。后续以当前环境扫描结果为准。

## 对当前数据集的取舍

第一版“无机器人、无主动外力”的物理常识数据集，不应该把全部 1,136 个模型都直接塞进去。更稳的顺序是：

1. **第一批**：原生几何体和 Kubric primitive  
   球、方块、圆柱、墙、斜面、遮挡板、柱体。

2. **第二批**：少量刚体 URDF  
   `soccerball.urdf`、`cube.urdf`、`block.urdf`、`lego/lego.urdf`、`duck_vhacd.urdf`、`teddy_vhacd.urdf`、`objects/mug.urdf`、`tray/tray.urdf`、`domino/domino.urdf`、`jenga/jenga.urdf`。

3. **第三批**：`random_urdfs/000-999` 抽样  
   先做批量加载、AABB 尺寸、质量、碰撞稳定性筛查，再纳入杂物/OOD 场景。

4. **暂缓**：机器人、车辆、四足、人形、MJCF 控制模型  
   这些更适合控制/机器人数据，不符合当前“无外部主动干预”的第一版目标。

## 分类明细

| 类别 | 条目数 | 展开计数 | 代表条目 |
|---|---:|---:|---|
| PyBullet 原生几何类型 | 7 | 7 | `GEOM_SPHERE`, `GEOM_BOX`, `GEOM_CYLINDER`, `GEOM_CAPSULE`, `GEOM_PLANE`, `GEOM_MESH`, `GEOM_HEIGHTFIELD` |
| 地面/场地 | 7 | 7 | `plane.urdf`, `stadium.sdf`, `plane_stadium.sdf` |
| 球体 | 9 | 9 | `sphere2.urdf`, `sphere_with_restitution.urdf`, `soccerball.urdf` |
| 方块/积木/块状物 | 9 | 9 | `cube.urdf`, `block.urdf`, `lego/lego.urdf`, `jenga/jenga.urdf`, `domino/domino.urdf` |
| 日常物体/容器/玩具 | 9 | 9 | `duck_vhacd.urdf`, `teddy_vhacd.urdf`, `objects/mug.urdf`, `tray/tray.urdf` |
| 桌子/货架/环境 | 3 | 3 | `table/table.urdf`, `kiva_shelf/model.sdf` |
| 机械臂 | 9 | 9 | `franka_panda/panda.urdf`, `kuka_iiwa/model.urdf`, `xarm/xarm6_with_gripper.urdf` |
| 夹爪 | 8 | 8 | `gripper/wsg50_one_motor_gripper.sdf` 等 |
| 移动机器人/车辆 | 5 | 5 | `r2d2.urdf`, `racecar/racecar.urdf`, `husky/husky.urdf` |
| 四足/多足机器人 | 10 | 10 | `a1/a1.urdf`, `aliengo/aliengo.urdf`, `laikago/laikago.urdf` |
| 人形/双足/机构 | 8 | 8 | `humanoid/humanoid.urdf`, `cartpole.urdf`, `pendulum5.urdf` |
| MJCF 强化学习模型 | 17 | 17 | `mjcf/pusher.xml`, `mjcf/reacher.xml`, `mjcf/ant.xml` |
| 软体/地形资源 | 4 | 4 | `cloth_z_up.urdf`, `torus_deform.urdf`, `heightmaps/`, `GEOM_HEIGHTFIELD` |
| 随机 URDF 物体库 | 1 | 1,000 | `random_urdfs/000/000.urdf` 到 `random_urdfs/999/999.urdf` |
| Kubric 兼容实现入口 | 3 | 3 | `kb.Cube`, `kb.Sphere`, `FileBasedObject + .urdf` |
