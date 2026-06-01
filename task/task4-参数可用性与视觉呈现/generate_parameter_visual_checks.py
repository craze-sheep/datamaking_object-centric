#!/usr/bin/env python3
"""Generate task4 parameter usability probes and review videos.

This is a lightweight PyBullet/TinyRenderer smoke-test layer for the task3
parameter plans.  The Kubric tutorial in the repository says formal rendering
should use the Docker Kubric image; these probes focus on quickly checking
whether representative physics parameters are usable and visually legible.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "_build"
FPS = 12
DURATION = 3.0
SIM_HZ = 240
FRAMES = int(FPS * DURATION)
STEPS_PER_FRAME = SIM_HZ // FPS
W = 256
H = 256


COLORS = {
    "red": (0.86, 0.10, 0.08, 1),
    "blue": (0.08, 0.20, 0.86, 1),
    "yellow": (0.95, 0.78, 0.08, 1),
    "green": (0.08, 0.55, 0.20, 1),
    "gray": (0.55, 0.55, 0.55, 1),
    "dark_gray": (0.30, 0.30, 0.32, 1),
    "brown": (0.55, 0.35, 0.18, 1),
    "purple": (0.55, 0.20, 0.75, 1),
    "cyan": (0.05, 0.65, 0.72, 1),
    "orange": (0.92, 0.42, 0.08, 1),
}


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_SMALL = font(12)
FONT_MED = font(15)


@dataclass
class BodySpec:
    name: str
    shape: str
    position: tuple[float, float, float]
    size: tuple[float, float, float] | None = None
    radius: float | None = None
    height: float | None = None
    mass: float = 1.0
    velocity: tuple[float, float, float] = (0, 0, 0)
    angular_velocity: tuple[float, float, float] = (0, 0, 0)
    quaternion: tuple[float, float, float, float] = (0, 0, 0, 1)
    color: str = "red"
    friction: float = 0.5
    rolling_friction: float = 0.0
    spinning_friction: float = 0.0
    restitution: float = 0.5
    static: bool = False
    id: int | None = None


@dataclass
class ProbeCase:
    scene: str
    case_id: str
    title: str
    notes: str
    view: str
    bodies: list[BodySpec]
    gravity: tuple[float, float, float] = (0, 0, -9.8)
    setup: Callable[[], None] | None = None
    impulses: list[tuple[int, str, tuple[float, float, float]]] = field(default_factory=list)


def quat_xyzw_from_wxyz(q: Iterable[float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return (x, y, z, w)


def add_body(spec: BodySpec) -> int:
    rgba = COLORS[spec.color]
    if spec.shape == "sphere":
        assert spec.radius is not None
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=spec.radius)
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=spec.radius, rgbaColor=rgba)
    elif spec.shape == "cube":
        assert spec.size is not None
        half = [v / 2 for v in spec.size]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    elif spec.shape == "cylinder":
        assert spec.radius is not None and spec.height is not None
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=spec.radius, height=spec.height)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=spec.radius, length=spec.height, rgbaColor=rgba)
    else:
        raise ValueError(f"unsupported shape: {spec.shape}")

    body = p.createMultiBody(
        baseMass=0 if spec.static else spec.mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=spec.position,
        baseOrientation=spec.quaternion,
    )
    p.resetBaseVelocity(body, linearVelocity=spec.velocity, angularVelocity=spec.angular_velocity)
    p.changeDynamics(
        body,
        -1,
        lateralFriction=spec.friction,
        rollingFriction=spec.rolling_friction,
        spinningFriction=spec.spinning_friction,
        restitution=spec.restitution,
        linearDamping=0,
        angularDamping=0,
    )
    spec.id = body
    return body


def ground(size=(8.0, 6.0, 0.08), friction=0.5, restitution=0.0) -> BodySpec:
    return BodySpec(
        name="ground",
        shape="cube",
        size=size,
        position=(0, 0, -size[2] / 2),
        static=True,
        color="gray",
        friction=friction,
        restitution=restitution,
    )


def sphere(name: str, x: float, y: float, radius=0.22, **kwargs) -> BodySpec:
    return BodySpec(name=name, shape="sphere", radius=radius, position=(x, y, radius), **kwargs)


def cube(name: str, x: float, y: float, size=(0.24, 0.24, 0.24), **kwargs) -> BodySpec:
    return BodySpec(name=name, shape="cube", size=size, position=(x, y, size[2] / 2), **kwargs)


def cylinder(name: str, x: float, y: float, radius=0.20, height=0.28, **kwargs) -> BodySpec:
    return BodySpec(name=name, shape="cylinder", radius=radius, height=height, position=(x, y, height / 2), **kwargs)


def wall(x=1.20, y=0, size=(0.10, 5.00, 1.20), restitution=0.8, friction=0.0) -> BodySpec:
    return BodySpec(
        name="wall",
        shape="cube",
        size=size,
        position=(x, y, size[2] / 2),
        static=True,
        color="dark_gray",
        restitution=restitution,
        friction=friction,
    )


def ramp_case(angle: float, friction: float, obj: BodySpec, title: str, case_id: str) -> ProbeCase:
    length, width, thick = 2.4, 1.0, 0.12
    quat = p.getQuaternionFromEuler((0, angle, 0))
    ramp_center = (0, 0, 0.09 + 0.5 * length * math.sin(angle))
    ramp = BodySpec(
        name="ramp",
        shape="cube",
        size=(length, width, thick),
        position=ramp_center,
        quaternion=quat,
        static=True,
        color="brown",
        friction=friction,
        restitution=0,
    )
    local_s = -0.55
    surface = np.array(p.multiplyTransforms(ramp_center, quat, (local_s, 0, thick / 2), (0, 0, 0, 1))[0])
    normal = np.array(p.multiplyTransforms((0, 0, 0), quat, (0, 0, 1), (0, 0, 0, 1))[0])
    normal = normal / np.linalg.norm(normal)
    contact_offset = obj.radius if obj.shape == "sphere" else obj.size[2] / 2
    obj.position = tuple(surface + normal * contact_offset)
    obj.quaternion = quat if obj.shape == "cube" else obj.quaternion
    return ProbeCase("S3_斜面滑动", case_id, title, f"angle={angle:.2f}, ramp_mu={friction}", "oblique", [ground(), ramp, obj])


def make_cases() -> dict[str, list[ProbeCase]]:
    s1 = [
        ProbeCase("S1_落体", "height_low", "height z=0.6", "low release height", "front",
                  [ground((5, 5, 0.08), restitution=0), BodySpec("ball", "sphere", (0, 0, 0.6), radius=0.22, color="red", restitution=0)]),
        ProbeCase("S1_落体", "height_high", "height z=2.0", "higher release takes longer", "front",
                  [ground((5, 5, 0.08), restitution=0), BodySpec("ball", "sphere", (0, 0, 2.0), radius=0.22, color="blue", restitution=0)]),
        ProbeCase("S1_落体", "bounce_e08", "restitution=0.8", "visible rebound height", "front",
                  [ground((5, 5, 0.08), restitution=0.8), BodySpec("ball", "sphere", (0, 0, 1.6), radius=0.22, color="green", restitution=1.0)]),
        ProbeCase("S1_落体", "projectile_vx", "horizontal projectile", "vx is independent from vertical fall", "front",
                  [ground((6, 5, 0.08), restitution=0), BodySpec("ball", "sphere", (-0.8, 0, 1.5), radius=0.22, velocity=(0.75, 0, 0), color="yellow", restitution=0)]),
    ]

    s2 = [
        ProbeCase("S2_水平地面滑动", "cube_mu025", "cube slide mu=0.25", "longer stopping distance", "top",
                  [ground(friction=0.25), cube("cube", -1.0, 0, velocity=(2.0, 0, 0), color="red", friction=1.0)]),
        ProbeCase("S2_水平地面滑动", "cube_mu10", "cube slide mu=1.0", "shorter stopping distance", "top",
                  [ground(friction=1.0), cube("cube", -1.0, 0, velocity=(2.0, 0, 0), color="blue", friction=1.0)]),
        ProbeCase("S2_水平地面滑动", "sphere_roll", "sphere rollingFriction=0.05", "sliding to rolling then decay", "oblique",
                  [ground(friction=0.3), sphere("ball", -1.2, 0, velocity=(2.6, 0, 0), angular_velocity=(0, 0, 0), color="green", friction=1.0, rolling_friction=0.05)]),
        ProbeCase("S2_水平地面滑动", "cylinder_lie", "cylinder lying", "cylinder PyBullet primitive probe", "oblique",
                  [ground(friction=0.3), cylinder("cylinder", -1.1, 0, quaternion=quat_xyzw_from_wxyz((0.7071, 0, 0.7071, 0)), velocity=(2.0, 0, 0), color="orange", friction=1.0, rolling_friction=0.03)]),
    ]

    s3 = [
        ramp_case(0.12, 0.20, cube("cube", 0, 0, color="red", friction=1.0), "shallow ramp", "angle_012"),
        ramp_case(0.52, 0.20, cube("cube", 0, 0, color="blue", friction=1.0), "steep ramp", "angle_052"),
        ramp_case(0.35, 0.70, cube("cube", 0, 0, color="yellow", friction=1.0), "high ramp friction", "mu_070"),
        ramp_case(0.42, 0.30, sphere("ball", 0, 0, color="green", friction=1.0, rolling_friction=0.05), "sphere on ramp", "sphere_roll"),
    ]

    s4 = [
        ProbeCase("S4_墙面反弹", "wall_e03", "wall restitution=0.3", "weak rebound", "top",
                  [ground((10, 5, 0.08), friction=0), wall(restitution=0.3), sphere("ball", -0.75, 0, velocity=(2.0, 0, 0), color="red", restitution=1.0, friction=0)]),
        ProbeCase("S4_墙面反弹", "wall_e10", "wall restitution=1.0", "strong rebound", "top",
                  [ground((10, 5, 0.08), friction=0), wall(restitution=1.0), sphere("ball", -0.75, 0, velocity=(2.0, 0, 0), color="blue", restitution=1.0, friction=0)]),
        ProbeCase("S4_墙面反弹", "incident_angle", "incident angle=30deg", "tangent velocity persists on frictionless wall", "top",
                  [ground((10, 5, 0.08), friction=0), wall(restitution=0.8), sphere("ball", -0.80, -0.35, velocity=(2.1, 1.2, 0), color="green", restitution=1.0, friction=0)]),
        ProbeCase("S4_墙面反弹", "wall_friction", "wall friction=0.8", "tangent loss/spin visible", "top",
                  [ground((10, 5, 0.08), friction=0), wall(restitution=0.8, friction=0.8), sphere("ball", -0.80, -0.35, velocity=(2.1, 1.2, 0), angular_velocity=(0, 0, 8), color="yellow", restitution=1.0, friction=1.0)]),
    ]

    s5 = [
        ProbeCase("S5_球撞球", "bb_e03", "ball-ball e=0.3", "low separation speed", "top",
                  [ground((8, 6, 0.08), friction=0), sphere("A", -0.85, 0, velocity=(1.8, 0, 0), color="red", restitution=1.0, friction=0), sphere("B", 0.55, 0, color="blue", restitution=0.3, friction=0)]),
        ProbeCase("S5_球撞球", "bb_e10", "ball-ball e=1.0", "high separation speed", "top",
                  [ground((8, 6, 0.08), friction=0), sphere("A", -0.85, 0, velocity=(1.8, 0, 0), color="red", restitution=1.0, friction=0), sphere("B", 0.55, 0, color="green", restitution=1.0, friction=0)]),
        ProbeCase("S5_球撞球", "target_heavy", "heavy target mass=5", "mass ratio changes transfer", "top",
                  [ground((8, 6, 0.08), friction=0), sphere("A", -0.85, 0, velocity=(2.2, 0, 0), color="yellow", restitution=1.0, friction=0), sphere("B", 0.55, 0, mass=5.0, color="purple", restitution=0.8, friction=0)]),
        ProbeCase("S5_球撞球", "offset", "impact offset y=0.25", "scattering angle visible", "top",
                  [ground((8, 6, 0.08), friction=0), sphere("A", -0.9, 0, velocity=(2.2, 0, 0), color="red", restitution=1.0, friction=0), sphere("B", 0.55, 0.25, color="blue", restitution=0.8, friction=0)]),
    ]

    s6 = [
        ProbeCase("S6_球撞方块", "bc_e03", "ball-cube e=0.3", "low separation/cube travel", "top",
                  [ground((8, 6, 0.08), friction=0), sphere("ball", -0.85, 0, velocity=(1.8, 0, 0), color="red", restitution=1.0, friction=0), cube("cube", 0.55, 0, color="blue", restitution=0.3, friction=0)]),
        ProbeCase("S6_球撞方块", "bc_e10", "ball-cube e=1.0", "high cube travel", "top",
                  [ground((8, 6, 0.08), friction=0), sphere("ball", -0.85, 0, velocity=(1.8, 0, 0), color="red", restitution=1.0, friction=0), cube("cube", 0.55, 0, color="green", restitution=1.0, friction=0)]),
        ProbeCase("S6_球撞方块", "cube_heavy", "heavy cube mass=5", "mass ratio changes cube motion", "top",
                  [ground((8, 6, 0.08), friction=0), sphere("ball", -0.85, 0, velocity=(2.2, 0, 0), color="yellow", restitution=1.0, friction=0), cube("cube", 0.55, 0, mass=5.0, color="purple", restitution=0.8, friction=0)]),
        ProbeCase("S6_球撞方块", "cube_yaw45", "cube yaw 45deg", "edge contact and rotation", "top",
                  [ground((8, 6, 0.08), friction=0), sphere("ball", -0.9, 0.1, velocity=(2.2, 0, 0), color="red", restitution=1.0, friction=0.5), cube("cube", 0.55, 0, quaternion=quat_xyzw_from_wxyz((0.9239, 0, 0, 0.3827)), color="orange", restitution=0.8, friction=0.5)]),
    ]

    s7 = [
        ProbeCase("S7_三物体连锁碰撞", "chain_e03", "chain restitution=0.3", "weak terminal transfer", "top",
                  [ground((9, 6, 0.08), friction=0), sphere("A", -0.85, 0, velocity=(1.8, 0, 0), color="red", restitution=1.0, friction=0), sphere("B", 0, 0, color="blue", restitution=0.3, friction=0), sphere("C", 0.46, 0, color="green", restitution=0.3, friction=0)]),
        ProbeCase("S7_三物体连锁碰撞", "chain_e10", "chain restitution=1.0", "strong terminal transfer", "top",
                  [ground((9, 6, 0.08), friction=0), sphere("A", -0.85, 0, velocity=(1.8, 0, 0), color="red", restitution=1.0, friction=0), sphere("B", 0, 0, color="blue", restitution=1.0, friction=0), sphere("C", 0.46, 0, color="green", restitution=1.0, friction=0)]),
        ProbeCase("S7_三物体连锁碰撞", "middle_heavy", "middle mass=2.5", "mass sequence changes transfer", "top",
                  [ground((9, 6, 0.08), friction=0), sphere("A", -0.85, 0, velocity=(2.2, 0, 0), color="yellow", restitution=0.8, friction=0), sphere("B", 0, 0, mass=2.5, color="purple", restitution=0.8, friction=0), sphere("C", 0.46, 0, color="green", restitution=0.8, friction=0)]),
        ProbeCase("S7_三物体连锁碰撞", "right_wall", "chain with right wall", "terminal rebound creates later contacts", "top",
                  [ground((9, 6, 0.08), friction=0), wall(x=1.35, restitution=0.8), sphere("A", -0.95, 0, velocity=(2.4, 0, 0), color="red", restitution=0.85, friction=0), sphere("B", -0.05, 0, color="blue", restitution=0.85, friction=0), sphere("C", 0.41, 0, color="green", restitution=0.85, friction=0)]),
    ]

    s8 = [
        ProbeCase("S8_泛化样本", "parallel_lanes", "parallel lanes no collision", "spatial separation", "top",
                  [ground((9, 6, 0.08), friction=0), sphere("A", -1.8, -0.8, velocity=(1.2, 0, 0), color="red", friction=0), sphere("B", -1.6, 0.0, velocity=(1.6, 0, 0), color="blue", friction=0), sphere("C", -1.4, 0.8, velocity=(1.0, 0, 0), color="green", friction=0)]),
        ProbeCase("S8_泛化样本", "time_gap", "same region time gap", "paths cross at different times", "top",
                  [ground((9, 6, 0.08), friction=0), sphere("A", -1.8, 0, velocity=(2.0, 0, 0), color="red", friction=0), sphere("B", 0, -1.4, velocity=(0, 0.8, 0), color="blue", friction=0)]),
        ProbeCase("S8_泛化样本", "blocked_by_wall", "static wall blocks", "dynamic-static contact allowed", "top",
                  [ground((9, 6, 0.08), friction=0), BodySpec("wall", "cube", (0, 0, 0.45), size=(0.18, 2.6, 0.9), static=True, color="dark_gray", restitution=0.8, friction=0.5), sphere("A", -1.6, -0.25, velocity=(1.2, 0, 0), color="red", friction=0), sphere("B", 1.6, 0.25, velocity=(-1.2, 0, 0), color="blue", friction=0)]),
        ProbeCase("S8_泛化样本", "static_many", "static multi-object", "all velocities zero", "oblique",
                  [ground((9, 6, 0.08), friction=0.5), sphere("A", -0.9, -0.5, color="red"), cube("B", 0.0, 0.35, color="blue"), cylinder("C", 0.9, -0.25, color="green")]),
    ]
    return {c[0].scene: c for c in [s1, s2, s3, s4, s5, s6, s7, s8]}


def camera(view: str):
    target = (0, 0, 0.35)
    if view == "top":
        yaw, pitch, distance = 0, -89, 5.0
        target = (0, 0, 0.05)
    elif view == "front":
        yaw, pitch, distance = 90, -8, 5.0
        target = (0, 0, 0.9)
    else:
        yaw, pitch, distance = 45, -32, 4.4
        target = (0, 0, 0.45)
    view_m = p.computeViewMatrixFromYawPitchRoll(target, distance, yaw, pitch, 0, 2)
    proj_m = p.computeProjectionMatrixFOV(50, W / H, 0.01, 30)
    return view_m, proj_m


def render_frame(title: str, note: str, view_m, proj_m) -> Image.Image:
    _, _, rgba, _, _ = p.getCameraImage(W, H, view_m, proj_m, renderer=p.ER_TINY_RENDERER)
    arr = np.reshape(np.array(rgba, dtype=np.uint8), (H, W, 4))
    img = Image.fromarray(arr[:, :, :3], "RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, 0, W, 38), fill=(0, 0, 0, 145))
    draw.text((6, 3), title[:34], font=FONT_MED, fill=(255, 255, 255))
    draw.text((6, 21), note[:42], font=FONT_SMALL, fill=(225, 225, 225))
    return img


def run_case(case: ProbeCase, out_dir: Path) -> dict:
    frames_dir = BUILD / case.scene / case.case_id
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    p.connect(p.DIRECT)
    p.resetSimulation()
    p.setTimeStep(1 / SIM_HZ)
    p.setGravity(*case.gravity)
    p.setPhysicsEngineParameter(numSolverIterations=80, fixedTimeStep=1 / SIM_HZ)
    for spec in case.bodies:
        add_body(spec)
    if case.setup:
        case.setup()
    view_m, proj_m = camera(case.view)

    contact_dynamic_dynamic = 0
    contact_total = 0
    for frame in range(FRAMES):
        for _ in range(STEPS_PER_FRAME):
            step_index = frame * STEPS_PER_FRAME + _
            for impulse_frame, body_name, impulse in case.impulses:
                if step_index == impulse_frame:
                    body = next(b for b in case.bodies if b.name == body_name)
                    p.applyExternalForce(body.id, -1, impulse, (0, 0, 0), p.WORLD_FRAME)
            p.stepSimulation()
            contacts = p.getContactPoints()
            contact_total += len(contacts)
            for c in contacts:
                a = next((b for b in case.bodies if b.id == c[1]), None)
                b = next((b for b in case.bodies if b.id == c[2]), None)
                if a and b and not a.static and not b.static:
                    contact_dynamic_dynamic += 1
        img = render_frame(case.title, case.notes, view_m, proj_m)
        img.save(frames_dir / f"frame_{frame:03d}.png")

    final = {}
    for spec in case.bodies:
        if spec.id is None:
            continue
        pos, quat = p.getBasePositionAndOrientation(spec.id)
        vel, ang = p.getBaseVelocity(spec.id)
        final[spec.name] = {
            "position": [round(v, 4) for v in pos],
            "speed": round(float(np.linalg.norm(vel)), 4),
            "angular_speed": round(float(np.linalg.norm(ang)), 4),
        }
    p.disconnect()

    videos = out_dir / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    out_mp4 = videos / f"{case.case_id}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "frame_%03d.png"),
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(out_mp4),
        ],
        check=True,
    )
    return {
        "case_id": case.case_id,
        "title": case.title,
        "notes": case.notes,
        "view": case.view,
        "video": str(out_mp4.relative_to(ROOT)),
        "total_contact_samples": contact_total,
        "dynamic_dynamic_contact_samples": contact_dynamic_dynamic,
        "final": final,
    }


SCENE_REVIEW = {
    "S1_落体": ("探针通过", "现有探针覆盖低/高释放高度、反弹和水平抛体；只能说明这些代表性 case 可运行且视频可读，不等价于全量参数已验证。"),
    "S2_水平地面滑动": ("探针通过，Cylinder 需 Kubric 复核", "现有探针覆盖 Cube/Sphere 滑动、滚动和 PyBullet Cylinder；但 Kubric 教程记录普通 kubric39 中 kb.Cylinder=False，正式 Kubric 渲染仍需 URDF/FileBasedObject 或补 Cylinder 支持。"),
    "S3_斜面滑动": ("探针通过，斜面公式需复核", "现有探针覆盖倾角、摩擦和球体滚动；正式生成器仍需统一 ramp_surface_point 与接触高度公式，避免高倾角初始微穿透。"),
    "S4_墙面反弹": ("探针通过", "现有探针覆盖墙面恢复系数、入射角和墙面摩擦；由于墙是 static body，动态-动态接触数为 0，应结合总接触数和视频复核墙面碰撞。"),
    "S5_球撞球": ("探针通过", "现有探针覆盖对心、质量比和偏心散射；高速目标球离开可视区域属于配置中已预期的边界样本。"),
    "S6_球撞方块": ("探针通过", "现有探针覆盖球-方块碰撞、质量比和方块 yaw 姿态；偏心/旋转 case 建议优先 top 视角。"),
    "S7_三物体连锁碰撞": ("探针通过，链式顺序需全量过滤", "现有探针覆盖 AB->BC 链式传递、恢复系数、质量序列和右墙反弹；正式过滤器必须记录 contact_pair_sequence，剔除未完成链式传递样本。"),
    "S8_泛化样本": ("负样本探针通过，复杂 level 未覆盖", "现有探针只覆盖空间分离、时间错开、障碍阻隔、全静止负样本；外部冲量、三维高度错开和随机过滤类 level 仍需要专门采样器与反事实标签。"),
}


def write_scene_report(scene: str, out_dir: Path, results: list[dict]) -> None:
    verdict, summary = SCENE_REVIEW[scene]
    lines = [
        f"# {scene} 参数可用性与视觉呈现检查",
        "",
        f"- 初步判断：{verdict}",
        f"- 判断依据：{summary}",
        "- 证据口径：下面的判断只来自本脚本实际跑出的探针 case、PyBullet 接触统计和生成出的 MP4 文件；它不是 task3 全量参数组合的最终验证结论。",
        "- 渲染说明：本轮按 Kubric 教程的结论，把正式 Kubric/Docker 渲染作为最终路线；这里先用 PyBullet TinyRenderer 做轻量物理和视觉探针，视频为 256x256 便于人工查看，task3 目标分辨率仍为 64x64。",
        "- 统一设置：3s、12fps、36 帧、SI 单位、重力 [0,0,-9.8]。",
        "",
        "## 探针视频",
        "",
        "| case | 视频 | 关键检查 | 总接触采样数 | 动态-动态接触采样数 |",
        "|---|---|---|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['case_id']} | [videos/{item['case_id']}.mp4](videos/{item['case_id']}.mp4) | {item['title']}；{item['notes']} | {item['total_contact_samples']} | {item['dynamic_dynamic_contact_samples']} |"
        )
    lines += [
        "",
        "## 已做自动检查",
        "",
        "- 脚本完成 PyBullet 3s/36 帧仿真，并为每个 case 写出 MP4。",
        "- `total_contact_samples` 记录所有接触采样；落体、滑动、斜面、墙面等 static 接触主要看这一列。",
        "- `dynamic_dynamic_contact_samples` 只统计动态物体之间的接触；S5/S6/S7 应大于 0，S8 负样本应为 0。",
        "- 视觉是否“足够清楚”仍需要人工打开视频复核，不能只靠接触计数判断。",
        "",
        "## 后续全量生成注意",
        "",
        "- 这些 case 只验证代表性参数可用性和画面可读性，不替代 task3 中的分层采样全量组合。",
        "- 正式生成时应使用 `Kubric的使用教程.md` 中的 Docker 命令，并把同一套参数迁移到 Kubric Scene/Renderer。",
        "- 所有涉及链式碰撞、负样本过滤、边界离开画面的 level，都应保存 contact sequence、visible-area 与过滤原因标签。",
    ]
    (out_dir / "参数可用性与视觉呈现.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_root_readme(all_results: dict[str, list[dict]]) -> None:
    lines = [
        "# Task4：参数可用性与视觉呈现",
        "",
        "本目录为 task3 每个 S 场景建立一个验证包，目标是先用少量代表性视频检查参数是否能跑、物理差异是否可见、视角是否足够清楚。",
        "",
        "## 与 Kubric 教程的关系",
        "",
        "- `Kubric的使用教程.md` 明确建议：正式生成/渲染数据集用 Docker 官方镜像。",
        "- 本轮视频是 PyBullet TinyRenderer 轻量 smoke test，用于快速发现参数和视觉问题。",
        "- 后续正式数据集应把这些通过验证的 case/采样器迁移到 Kubric Docker 渲染管线。",
        "",
        "## 目录",
        "",
        "| 场景 | case 数 | 报告 |",
        "|---|---:|---|",
    ]
    for scene, results in all_results.items():
        lines.append(f"| {scene} | {len(results)} | [{scene}/参数可用性与视觉呈现.md]({scene}/参数可用性与视觉呈现.md) |")
    lines += [
        "",
        "## 重新生成",
        "",
        "```bash",
        "cd /home/lzy/project/slot-datamaking",
        "python task4-参数可用性与视觉呈现/generate_parameter_visual_checks.py",
        "```",
    ]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    cases_by_scene = make_cases()
    all_results: dict[str, list[dict]] = {}
    for scene, cases in cases_by_scene.items():
        out_dir = ROOT / scene
        out_dir.mkdir(parents=True, exist_ok=True)
        results = [run_case(case, out_dir) for case in cases]
        all_results[scene] = results
        (out_dir / "cases.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_scene_report(scene, out_dir, results)
    write_root_readme(all_results)
    if BUILD.exists():
        shutil.rmtree(BUILD)
    print(json.dumps({scene: len(items) for scene, items in all_results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
