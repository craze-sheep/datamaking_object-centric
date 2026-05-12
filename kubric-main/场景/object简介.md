# PyBullet / pybullet_data 可用物体详细简介

> 本文基于当前环境中实际安装的 `pybullet` 与 `pybullet_data` 扫描结果整理。  
> `pybullet_data` 路径：`/home/lzy/.local/lib/python3.13/site-packages/pybullet_data`

---

## 1. 先分清：PyBullet 里“物体”来自哪里

PyBullet 本身不是一个固定物体库。它提供的是物理引擎和几种创建/加载物体的方式：

| 来源 | 方式 | 典型物体 | 适合用途 |
|------|------|----------|----------|
| 原生几何体 | `createCollisionShape` + `createMultiBody` | 球、方块、圆柱、胶囊、平面、高度场、mesh | 快速构造可控物理实验 |
| URDF | `loadURDF()` | 机器人、桌子、托盘、杯子、球、方块、车、机械臂 | 机器人仿真和标准示例 |
| SDF | `loadSDF()` | KUKA+夹爪、场地、货架、夹爪系统 | 多物体/环境组合 |
| MJCF XML | `loadMJCF()` | Ant、Humanoid、HalfCheetah、Hopper 等强化学习模型 | 控制任务和MuJoCo风格模型 |
| Mesh | OBJ/STL/DAE，经URDF或直接mesh collision shape使用 | 复杂外观/碰撞几何 | 真实外观和复杂形状 |
| Soft body | `loadSoftBody()` 或相关示例资源 | cloth、可变形torus等 | 软体/布料探索 |

当前安装包扫描结果：

| 资源类型 | 数量 |
|----------|------|
| URDF | 1095 |
| SDF | 15 |
| MJCF XML | 26 |
| OBJ mesh | 1117 |
| STL mesh | 71 |
| DAE mesh | 12 |

注意：1095个URDF里有1000个来自 `random_urdfs/000` 到 `random_urdfs/999`，它们是随机几何组合物体，适合做杂物/干扰物。

---

## 2. PyBullet 原生几何体

这些是 PyBullet 的基础碰撞几何，可以不依赖任何文件直接创建。

| 常量 | 中文名 | 典型参数 | 说明 |
|------|--------|----------|------|
| `p.GEOM_SPHERE` | 球体 | `radius` | 小球、弹珠、滚动物体 |
| `p.GEOM_BOX` | 方块/长方体 | `halfExtents=[x,y,z]` | 方块、墙、桌面、推杆、障碍物 |
| `p.GEOM_CYLINDER` | 圆柱 | `radius`, `height` | 柱体、轮子、圆桶 |
| `p.GEOM_CAPSULE` | 胶囊体 | `radius`, `height` | 机器人连杆、简化人物肢体 |
| `p.GEOM_PLANE` | 无限平面 | 无或法向参数 | 地面 |
| `p.GEOM_MESH` | 三角网格 | `fileName`, `meshScale` | 任意OBJ/STL mesh |
| `p.GEOM_HEIGHTFIELD` | 高度场 | `heightfieldData` / 文件 | 地形、起伏表面 |

PyBullet 里还暴露了一些和 mesh 处理相关的标志：

| 常量 | 说明 |
|------|------|
| `p.GEOM_FORCE_CONCAVE_TRIMESH` | 强制使用凹三角网格，常用于静态复杂场景 |
| `p.GEOM_CONCAVE_INTERNAL_EDGE` | 凹mesh内部边处理相关标志 |

### 2.1 直接创建球和方块

```python
import pybullet as p

p.connect(p.GUI)  # 或 p.DIRECT
p.setGravity(0, 0, -9.81)

# 地面
plane_col = p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_col)

# 球
sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
sphere_id = p.createMultiBody(
    baseMass=1.0,
    baseCollisionShapeIndex=sphere_col,
    basePosition=[0, 0, 1],
)

# 方块
box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
box_id = p.createMultiBody(
    baseMass=1.0,
    baseCollisionShapeIndex=box_col,
    basePosition=[1, 0, 1],
)
```

### 2.2 推荐用于本项目的原生形状

| 项目需求 | 推荐形状 | 原因 |
|----------|----------|------|
| 小球 | `GEOM_SPHERE` | 物理稳定、滚动自然、碰撞简单 |
| 方块/目标块 | `GEOM_BOX` | 最稳定、最适合做位移/碰撞结果标注 |
| 推杆/机械臂末端代理 | `GEOM_BOX` 或 `GEOM_CYLINDER` | 比完整机械臂更容易控制 |
| 障碍物 | `GEOM_BOX`、`GEOM_CYLINDER` | 容易参数化位置和尺寸 |
| 桌面/墙面 | `GEOM_BOX` 或 `GEOM_PLANE` | 可做边界和支撑 |
| 复杂真实物体 | `GEOM_MESH` 或 URDF mesh | 适合后期提升视觉复杂度 |

---

## 3. pybullet_data 中的基础物体

这些是可以直接 `loadURDF()` 的常用物体。

### 3.1 地面/场地

| 文件 | 说明 |
|------|------|
| `plane.urdf` | 标准平面地面 |
| `plane100.urdf` | 大面积地面 |
| `plane_implicit.urdf` | 隐式平面 |
| `plane_transparent.urdf` | 透明平面 |
| `stadium.sdf` | 体育场/场地环境 |
| `stadium_no_collision.sdf` | 无碰撞场地 |
| `plane_stadium.sdf` | 平面+场地组合 |

示例：

```python
import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
```

### 3.2 球体

| 文件 | 说明 |
|------|------|
| `sphere2.urdf` | 标准球 |
| `sphere2red.urdf` | 红色球 |
| `sphere2red_nocol.urdf` | 红色球，无碰撞版本 |
| `sphere_1cm.urdf` | 1cm小球 |
| `sphere_small.urdf` | 小球 |
| `sphere_transparent.urdf` | 透明球 |
| `sphere_with_restitution.urdf` | 带恢复系数设置的球 |
| `soccerball.urdf` | 足球外观球体 |
| `sphere8cube.urdf` | 球/方块组合示例 |

对碰撞数据集而言，建议优先使用原生 `GEOM_SPHERE` 或 `sphere_with_restitution.urdf`，这样恢复系数和质量更容易控制。

### 3.3 方块/积木/块状物

| 文件 | 说明 |
|------|------|
| `cube.urdf` | 标准方块 |
| `cube_small.urdf` | 小方块 |
| `cube_no_rotation.urdf` | 禁止旋转的方块 |
| `cube_rotate.urdf` | 可旋转方块 |
| `cube_collisionfilter.urdf` | 碰撞过滤示例方块 |
| `block.urdf` | 块状物 |
| `lego/lego.urdf` | 乐高块 |
| `jenga/jenga.urdf` | Jenga木块 |
| `domino/domino.urdf` | 多米诺骨牌 |

用途建议：

| 任务 | 推荐 |
|------|------|
| 球撞方块 | `cube.urdf` 或原生 `GEOM_BOX` |
| 堆叠稳定性 | `jenga/jenga.urdf` |
| 连锁倒塌 | `domino/domino.urdf` |
| 杂物碰撞 | `lego/lego.urdf` |

### 3.4 日常物体/容器/玩具

| 文件 | 说明 |
|------|------|
| `duck_vhacd.urdf` | 经典小黄鸭，带VHACD碰撞近似 |
| `teddy_large.urdf` | 大泰迪熊 |
| `teddy_vhacd.urdf` | 泰迪熊VHACD碰撞版本 |
| `objects/mug.urdf` | 杯子 |
| `urdf/mug.urdf` | 杯子另一路径版本 |
| `tray/tray.urdf` | 托盘 |
| `tray/tray_textured2.urdf` | 带纹理托盘 |
| `tray/traybox.urdf` | 盒式托盘 |
| `toys/concave_box.urdf` | 凹形玩具盒 |

对“推球碰撞方块”项目，这些更适合第二阶段加入为干扰物或真实外观目标，不建议第一版就混入太多复杂mesh。

### 3.5 桌子/货架/环境物体

| 文件 | 说明 |
|------|------|
| `table/table.urdf` | 桌子 |
| `table_square/table_square.urdf` | 方桌 |
| `kiva_shelf/model.sdf` | Kiva仓储货架 |

用途：
- 桌面机器人场景。
- 货架/仓储场景。
- 作为静态背景和碰撞边界。

---

## 4. 机器人和可动机构

pybullet_data 里有很多机器人模型，适合做控制、抓取、移动、四足行走等任务。

### 4.1 机械臂

| 机器人 | 文件 | 说明 | 对本项目价值 |
|--------|------|------|--------------|
| Franka Panda | `franka_panda/panda.urdf` | 7自由度协作机械臂，带手爪mesh | 很高，最贴近DROID/真实机器人数据 |
| KUKA iiwa | `kuka_iiwa/model.urdf` | 7自由度机械臂 | 很高，PyBullet经典示例 |
| KUKA free base | `kuka_iiwa/model_free_base.urdf` | 自由基座版本 | 用于特殊控制实验 |
| KUKA VR limits | `kuka_iiwa/model_vr_limits.urdf` | 带VR/控制限制版本 | 控制实验 |
| KUKA + gripper | `kuka_iiwa/kuka_with_gripper.sdf` | 机械臂+夹爪组合 | 抓取/推动 |
| xArm6 | `xarm/xarm6_robot.urdf` | 6自由度机械臂 | 很高，可做机械臂推物 |
| xArm6 white | `xarm/xarm6_robot_white.urdf` | 白色外观版本 | 渲染更清晰 |
| xArm6 + gripper | `xarm/xarm6_with_gripper.urdf` | xArm6加夹爪 | 抓取/推物 |
| PR2 gripper | `pr2_gripper.urdf` | PR2夹爪 | 单独夹爪实验 |

机械臂加载示例：

```python
import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

robot_id = p.loadURDF(
    "franka_panda/panda.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True,
)

num_joints = p.getNumJoints(robot_id)
for j in range(num_joints):
    print(j, p.getJointInfo(robot_id, j)[1])
```

对你的“机械臂推动小球”任务，路线建议：

1. 第一版：不要直接上完整机械臂，先用原生 `GEOM_BOX` 做推杆，控制稳定。
2. 第二版：使用 `franka_panda/panda.urdf` 或 `xarm/xarm6_with_gripper.urdf`，用末端执行器推动球。
3. 第三版：加入逆运动学 `calculateInverseKinematics()` 和关节控制 `setJointMotorControl2()`。

### 4.2 夹爪

| 文件 | 说明 |
|------|------|
| `gripper/wsg50_one_motor_gripper.sdf` | WSG50 单电机夹爪 |
| `gripper/wsg50_one_motor_gripper_free_base.sdf` | 自由基座夹爪 |
| `gripper/wsg50_one_motor_gripper_new.sdf` | 新版WSG50夹爪 |
| `gripper/wsg50_one_motor_gripper_new_free_base.sdf` | 新版自由基座夹爪 |
| `gripper/wsg50_one_motor_gripper_no_finger.sdf` | 无手指版本 |
| `gripper/wsg50_with_r2d2_gripper.sdf` | WSG50 + R2D2组合 |
| `gripper/wsg50_one_motor_gripper_left_finger.urdf` | 左指 |
| `gripper/wsg50_one_motor_gripper_right_finger.urdf` | 右指 |

夹爪适合做抓取、夹持、推挤，但单独用会比“推杆代理”更复杂。

### 4.3 移动机器人/车辆

| 机器人 | 文件 | 说明 |
|--------|------|------|
| R2D2 | `r2d2.urdf` | 经典移动机器人示例 |
| Racecar | `racecar/racecar.urdf` | 小车模型 |
| Racecar differential | `racecar/racecar_differential.urdf` | 差速/差分版本 |
| Husky | `husky/husky.urdf` | Clearpath Husky地面机器人 |
| Bicycle | `bicycle/bike.urdf` | 自行车模型 |

用途：
- 导航、避障、移动碰撞。
- 车辆推动/撞击物体。
- 不如机械臂适合桌面推球，但可以做“移动机器人撞球/推箱子”。

### 4.4 四足/多足机器人

| 机器人 | 文件 |
|--------|------|
| A1 | `a1/a1.urdf` |
| AlienGo | `aliengo/aliengo.urdf` |
| Laikago | `laikago/laikago.urdf` |
| Laikago toes | `laikago/laikago_toes.urdf` |
| Laikago toes limits | `laikago/laikago_toes_limits.urdf` |
| Mini Cheetah | `mini_cheetah/mini_cheetah.urdf` |
| Minitaur | `quadruped/minitaur.urdf` |
| Microtaur | `quadruped/microtaur/microtaur.urdf` |
| Spirit40 | `quadruped/spirit40.urdf` |
| Vision60 | `quadruped/vision60.urdf` |

用途：
- 强化学习行走。
- 复杂接触/摩擦。
- 对当前球-方块碰撞项目不是优先对象。

### 4.5 人形/双足/简单机构

| 文件 | 说明 |
|------|------|
| `humanoid/humanoid.urdf` | 人形机器人 |
| `biped/biped2d_pybullet.urdf` | 2D双足 |
| `TwoJointRobot_w_fixedJoints.urdf` | 两关节机器人，含固定关节 |
| `TwoJointRobot_wo_fixedJoints.urdf` | 两关节机器人，无固定关节 |
| `pendulum5.urdf` | 五连杆摆 |
| `cartpole.urdf` | 倒立摆/小车杆 |
| `spherical_joint_limit.urdf` | 球关节限制示例 |
| `differential/diff_ring.urdf` | 差动环机构 |

用途：
- 关节控制、动力学教学。
- 强化学习控制基准。
- 不适合作为物体碰撞数据集主线，但可作为复杂动力学扩展。

---

## 5. MJCF 强化学习模型

`pybullet_data/mjcf/` 下有 MuJoCo 风格 XML 模型，可以用 `p.loadMJCF()` 加载。

| 文件 | 说明 |
|------|------|
| `mjcf/ant.xml` | Ant四足控制任务 |
| `mjcf/half_cheetah.xml` | HalfCheetah |
| `mjcf/hopper.xml` | Hopper单腿跳跃 |
| `mjcf/humanoid.xml` | Humanoid |
| `mjcf/humanoid_fixed.xml` | 固定版本Humanoid |
| `mjcf/humanoid_symmetric.xml` | 对称Humanoid |
| `mjcf/walker2d.xml` | Walker2D |
| `mjcf/swimmer.xml` | Swimmer |
| `mjcf/reacher.xml` | Reacher机械臂任务 |
| `mjcf/pusher.xml` | Pusher推物任务 |
| `mjcf/striker.xml` | Striker击打任务 |
| `mjcf/thrower.xml` | Thrower投掷任务 |
| `mjcf/inverted_pendulum.xml` | 倒立摆 |
| `mjcf/inverted_double_pendulum.xml` | 双倒立摆 |
| `mjcf/capsule*.xml` | 胶囊形状示例 |
| `mjcf/cylinder*.xml` | 圆柱形状示例 |
| `mjcf/ground*.xml` | 地面示例 |

对当前项目最值得关注的是：

| 模型 | 价值 |
|------|------|
| `mjcf/pusher.xml` | 和“推物体”任务最像，可以参考动作/状态设计 |
| `mjcf/reacher.xml` | 可参考简单机械臂控制 |
| `mjcf/striker.xml` | 可参考击打/碰撞任务 |
| `mjcf/thrower.xml` | 可参考动力学事件任务 |

不过，MJCF 模型更偏控制任务，不一定适合直接接入 Kubric 的渲染/标注流程。

---

## 6. 软体、布料、可变形与地形

### 6.1 布料/软体

| 文件 | 说明 |
|------|------|
| `cloth_z_up.urdf` | 布料/软体相关示例资源 |
| `torus_deform.urdf` | 可变形torus示例 |

PyBullet 支持软体仿真相关接口，但它和刚体URDF流程不同，稳定性和标注复杂度也更高。对于第一版物理数据集，不建议优先使用软体。

### 6.2 高度场/地形

| 资源 | 说明 |
|------|------|
| `heightmaps/` | 高度图资源目录 |
| `p.GEOM_HEIGHTFIELD` | 可通过高度数据创建起伏地形 |

适合：
- 球在不平地形上滚动。
- 移动机器人越障。
- 摩擦/坡度/接触复杂性测试。

---

## 7. 随机物体库：`random_urdfs`

`random_urdfs` 是 pybullet_data 里数量最大的物体集合：

```text
random_urdfs/000/000.urdf
random_urdfs/001/001.urdf
...
random_urdfs/999/999.urdf
```

特点：
- 共 1000 个随机生成的URDF物体。
- 每个目录通常包含一个URDF和对应OBJ mesh。
- 适合作为随机干扰物、杂物、背景物体。

示例：

```python
obj_id = p.loadURDF(
    "random_urdfs/042/042.urdf",
    basePosition=[0.5, 0, 1],
    globalScaling=0.5,
)
```

用途建议：

| 用途 | 是否推荐 |
|------|----------|
| 增加场景杂乱度 | 推荐 |
| 做OOD测试物体 | 推荐 |
| 第一版核心物理推理目标 | 不推荐 |
| 精确可控质量/形状实验 | 不推荐 |

---

## 8. 关节类型

机器人和可动机构依赖关节。当前 PyBullet 暴露的主要关节类型：

| 常量 | 中文名 | 用途 |
|------|--------|------|
| `p.JOINT_REVOLUTE` | 转动关节 | 机械臂、轮子、铰链 |
| `p.JOINT_PRISMATIC` | 平移关节 | 滑轨、伸缩机构 |
| `p.JOINT_SPHERICAL` | 球关节 | 肩关节、复杂摆动 |
| `p.JOINT_PLANAR` | 平面关节 | 平面移动 |
| `p.JOINT_FIXED` | 固定关节 | 固定连接 |
| `p.JOINT_POINT2POINT` | 点到点约束 | 约束/连接 |
| `p.JOINT_GEAR` | 齿轮约束 | 联动机构 |

常用控制接口：

```python
p.setJointMotorControl2(
    bodyUniqueId=robot_id,
    jointIndex=joint_id,
    controlMode=p.POSITION_CONTROL,
    targetPosition=0.5,
    force=100,
)
```

---

## 9. Kubric 中的兼容情况

当前 Kubric 的 PyBullet 接口位于：

```text
kubric-main/kubric/simulator/pybullet.py
```

它直接支持：

| Kubric对象 | PyBullet实现 | 说明 |
|------------|--------------|------|
| `kb.Cube` | `p.GEOM_BOX` | 支持非均匀scale，适合地面/墙/方块/推杆 |
| `kb.Sphere` | `p.GEOM_SPHERE` | 要求三轴scale相同 |
| `kb.FileBasedObject` + `.urdf` | `p.loadURDF()` | 支持加载URDF文件 |

Kubric 当前这个接口不直接注册 `Cylinder`、`Capsule` 等类。如果要在 Kubric 内使用圆柱/胶囊，有三种办法：

1. 用 URDF 包装圆柱/胶囊，然后作为 `FileBasedObject` 加载。
2. 修改 `kubric/simulator/pybullet.py`，增加 `Cylinder` 类和 `p.GEOM_CYLINDER` 注册。
3. 用 mesh/asset source 导入，并为它准备对应URDF碰撞文件。

对当前项目的结论：

| 需求 | Kubric推荐实现 |
|------|----------------|
| 球 | `kb.Sphere` |
| 方块 | `kb.Cube` |
| 桌面/墙 | `kb.Cube(static=True)` |
| 推杆 | `kb.Cube`，通过关键帧或速度控制 |
| 真实机械臂 | 先在纯PyBullet里验证URDF；若要Kubric渲染，需要处理机器人mesh/URDF导入 |
| 杯子/托盘/杂物 | `FileBasedObject` + URDF，或使用 Kubric asset source |

---

## 10. 面向“机械臂推球碰撞方块”的物体选择建议

### 10.1 第一版：最稳、最可控

| 角色 | 物体 | 实现 |
|------|------|------|
| 地面/桌面 | 方形桌面 | `GEOM_BOX` 或 `kb.Cube(static=True)` |
| 推杆/末端执行器代理 | 长方体 | `GEOM_BOX` 或 `kb.Cube` |
| 被推小球 | 球体 | `GEOM_SPHERE` 或 `kb.Sphere` |
| 被撞目标 | 方块 | `GEOM_BOX` 或 `kb.Cube` |
| 障碍物 | 方块/圆柱 | `GEOM_BOX`/`GEOM_CYLINDER`，Kubric里优先方块 |

优点：
- 物理稳定。
- 标注清晰。
- 容易生成因果链：`push -> ball moves -> collision -> block moves`。

### 10.2 第二版：加入真实机器人

| 机器人 | 推荐程度 | 文件 |
|--------|----------|------|
| Franka Panda | 最高 | `franka_panda/panda.urdf` |
| KUKA iiwa | 高 | `kuka_iiwa/model.urdf` |
| xArm6 + gripper | 高 | `xarm/xarm6_with_gripper.urdf` |

推荐先用 PyBullet 验证控制，再考虑和 Kubric 渲染/标注打通。

### 10.3 第三版：加入真实杂物和OOD测试

| 类别 | 推荐文件 |
|------|----------|
| 杯子 | `objects/mug.urdf` |
| 托盘 | `tray/tray.urdf` |
| 乐高 | `lego/lego.urdf` |
| 多米诺 | `domino/domino.urdf` |
| Jenga | `jenga/jenga.urdf` |
| 随机杂物 | `random_urdfs/*/*.urdf` |

---

## 11. 快速索引：常用文件路径

### 基础物体

```text
plane.urdf
cube.urdf
cube_small.urdf
block.urdf
sphere2.urdf
sphere2red.urdf
sphere_with_restitution.urdf
soccerball.urdf
duck_vhacd.urdf
lego/lego.urdf
jenga/jenga.urdf
domino/domino.urdf
objects/mug.urdf
tray/tray.urdf
table/table.urdf
```

### 机器人

```text
franka_panda/panda.urdf
kuka_iiwa/model.urdf
kuka_iiwa/kuka_with_gripper.sdf
xarm/xarm6_robot.urdf
xarm/xarm6_with_gripper.urdf
r2d2.urdf
racecar/racecar.urdf
husky/husky.urdf
laikago/laikago.urdf
a1/a1.urdf
aliengo/aliengo.urdf
mini_cheetah/mini_cheetah.urdf
humanoid/humanoid.urdf
```

### 强化学习/MJCF

```text
mjcf/reacher.xml
mjcf/pusher.xml
mjcf/striker.xml
mjcf/thrower.xml
mjcf/ant.xml
mjcf/half_cheetah.xml
mjcf/hopper.xml
mjcf/humanoid.xml
mjcf/walker2d.xml
```

---

## 12. 一句话总结

PyBullet 里可用物体可以分成两层：**原生几何体**负责稳定、可控的物理实验，**pybullet_data 的 URDF/SDF/MJCF/mesh 资源**负责机器人、真实物体和复杂场景。对本项目来说，第一版应优先使用 `球 + 方块 + 推杆 + 桌面` 的原生几何体组合；等事件标注和数据流程稳定后，再接入 `Franka Panda / KUKA / xArm` 等机器人URDF。
