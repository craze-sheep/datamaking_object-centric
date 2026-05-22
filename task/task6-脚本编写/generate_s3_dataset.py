#!/usr/bin/env python3
"""Generate S3 horizontal sliding samples with Kubric.

Run inside the official Kubric Docker image from the repository root, for example:

  docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "/home/lzy/project/slot-datamaking:/workspace" \
    --volume "/home/lzy/project/slot-datamaking/kubric-main:/kubric" \
    --workdir /workspace \
    kubricdockerhub/kubruntu \
    /usr/bin/python3 task/task6-脚本编写/generate_s2_dataset.py --levels 1 --samples_per_level 2
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import numbers
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

if Path("/kubric").exists() and "/kubric" not in sys.path:
    sys.path.insert(0, "/kubric")
LOCAL_KUBRIC = Path(__file__).resolve().parents[2] / "kubric"
if LOCAL_KUBRIC.exists() and str(LOCAL_KUBRIC) not in sys.path:
    sys.path.insert(0, str(LOCAL_KUBRIC))
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import blender_argv_fix  # noqa: E402  — must come before argparse

import numpy as np
from physics_label_utils import compute_physics_labels

imageio = None
kb = None
Blender = None
PyBullet = None


FPS = 12
NUM_FRAMES = 36
SIM_HZ = 240
DURATION_S = 3.0
RESOLUTION = 128
GRAVITY = (0.0, 0.0, -9.8)
SCENE_ID = 3
GROUND_OBJECT_ID = 1
PRIMARY_OBJECT_ID = 2
RAMP_OBJECT_ID = 3
LEVEL_TARGETS = {1: 80, 2: 80, 3: 80, 4: 80, 5: 80, 6: 120, 7: 120, 8: 120, 9: 120}
VIEWS = {
    "front": {
        "type": "Perspective",
        "position": (0.0, -7.5, 3.2),
        "look_at": (0.0, 0.0, 0.35),
        "focal_length": 35,
        "sensor_width": 32,
    },
    "top": {
        "type": "Orthographic",
        "position": (0.0, -0.01, 8.0),
        "look_at": (0.0, 0.0, 0.0),
        "orthographic_scale": 5.0,
    },
    "left": {
        "type": "Perspective",
        "position": (-7.5, 0.0, 3.2),
        "look_at": (0.0, 0.0, 0.35),
        "focal_length": 35,
        "sensor_width": 32,
    },
}

COLORS = {
    "gray": (0.55, 0.55, 0.55, 1.0),
    "red": (0.88, 0.10, 0.08, 1.0),
    "blue": (0.08, 0.20, 0.85, 1.0),
    "yellow": (0.95, 0.80, 0.08, 1.0),
    "green": (0.12, 0.55, 0.20, 1.0),
    "brown": (0.45, 0.30, 0.15, 1.0),
}


@dataclass(frozen=True)
class ObjectSpec:
    object_id: int
    object_type: str
    name: str
    static: bool
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float | None = None
    size: tuple[float, float, float] | None = None
    height: float | None = None
    mass: float | None = None
    lateral_friction: float = 0.4
    rolling_friction: float = 0.0
    spinning_friction: float = 0.0
    restitution: float = 0.0
    color_name: str = "red"


@dataclass(frozen=True)
class SampleSpec:
    level_id: int
    sample_id: int
    level_name: str
    subtask: str
    main_variable: str
    objects: tuple[ObjectSpec, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="database")
    parser.add_argument("--levels", nargs="+", type=int, default=[1])
    parser.add_argument("--samples_per_level", type=int, default=None)
    parser.add_argument("--start_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--views", nargs="+", choices=sorted(VIEWS), default=["front", "top", "left"])
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--samples_per_pixel", type=int, default=32)
    parser.add_argument("--keep_scratch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_runtime_dependencies() -> None:
    global imageio, kb, Blender, PyBullet
    if imageio is not None:
        return
    import imageio.v2 as imageio_module
    import kubric as kb_module
    from kubric.renderer.blender import Blender as BlenderClass
    from kubric.simulator import PyBullet as PyBulletClass

    imageio = imageio_module
    kb = kb_module
    Blender = BlenderClass
    PyBullet = PyBulletClass


def json_default(value: Any) -> Any:
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)


def material(color_name: str, roughness: float = 0.55) -> kb.PrincipledBSDFMaterial:
    return kb.PrincipledBSDFMaterial(color=COLORS[color_name], roughness=roughness)


def ground_spec(restitution: float = 0.0, friction: float = 0.5) -> ObjectSpec:
    size = (4.0, 4.0, 0.08)
    return ObjectSpec(
        object_id=GROUND_OBJECT_ID,
        object_type="ground",
        name="ground",
        static=True,
        size=size,
        position=(0.0, 0.0, -size[2] / 2.0),
        lateral_friction=friction,
        rolling_friction=0.005,
        spinning_friction=0.001,
        restitution=restitution,
        color_name="gray",
    )


def cube_spec(
    object_id: int,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    mass: float = 1.0,
    restitution: float = 0.0,
    lateral_friction: float = 0.4,
    rolling_friction: float = 0.0,
    spinning_friction: float = 0.0,
    color_name: str = "red",
) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        object_type="cube",
        name=f"cube_{object_id}",
        static=False,
        size=size,
        position=position,
        quaternion=quaternion,
        velocity=velocity,
        angular_velocity=angular_velocity,
        mass=mass,
        lateral_friction=lateral_friction,
        rolling_friction=rolling_friction,
        spinning_friction=spinning_friction,
        restitution=restitution,
        color_name=color_name,
    )


def sphere_spec(
    object_id: int,
    radius: float,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    mass: float = 1.0,
    restitution: float = 0.0,
    lateral_friction: float = 0.4,
    rolling_friction: float = 0.0,
    spinning_friction: float = 0.0,
    color_name: str = "red",
) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        object_type="sphere",
        name=f"sphere_{object_id}",
        static=False,
        radius=radius,
        position=position,
        quaternion=quaternion,
        velocity=velocity,
        angular_velocity=angular_velocity,
        mass=mass,
        lateral_friction=lateral_friction,
        rolling_friction=rolling_friction,
        spinning_friction=spinning_friction,
        restitution=restitution,
        color_name=color_name,
    )


def cylinder_spec(
    object_id: int,
    radius: float,
    height: float,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    mass: float = 1.0,
    restitution: float = 0.0,
    lateral_friction: float = 0.4,
    rolling_friction: float = 0.0,
    spinning_friction: float = 0.0,
    color_name: str = "red",
) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        object_type="cylinder",
        name=f"cylinder_{object_id}",
        static=False,
        radius=radius,
        height=height,
        position=position,
        quaternion=quaternion,
        velocity=velocity,
        angular_velocity=angular_velocity,
        mass=mass,
        lateral_friction=lateral_friction,
        rolling_friction=rolling_friction,
        spinning_friction=spinning_friction,
        restitution=restitution,
        color_name=color_name,
    )



def ramp_spec(
    object_id: int,
    size: tuple[float, float, float],
    incline_angle: float,
    lateral_friction: float = 0.2,
    rolling_friction: float = 0.0,
    spinning_friction: float = 0.0,
    restitution: float = 0.0,
    color_name: str = "brown",
) -> tuple[ObjectSpec, float]:
    """Create a ramp spec. Returns (spec, incline_angle)."""
    half_angle = incline_angle / 2.0
    quat = (math.cos(half_angle), 0.0, math.sin(half_angle), 0.0)
    z = size[2] + 0.5 * size[0] * math.sin(incline_angle)
    position = (0.0, 0.0, z)
    spec = ObjectSpec(
        object_id=object_id,
        object_type="ramp",
        name="ramp",
        static=True,
        size=size,
        position=position,
        quaternion=quat,
        mass=1.0,
        lateral_friction=lateral_friction,
        rolling_friction=rolling_friction,
        spinning_friction=spinning_friction,
        restitution=restitution,
        color_name=color_name,
    )
    return spec, incline_angle


def ramp_surface_point(s: float, y: float, incline_angle: float, ramp_size: tuple[float, float, float]) -> tuple[float, float, float]:
    """Calculate a point on the ramp surface.
    s: distance along ramp from center (positive = downhill)
    y: lateral offset
    """
    cos_a = math.cos(incline_angle)
    sin_a = math.sin(incline_angle)
    x = s * cos_a
    z = ramp_size[2] + 0.5 * ramp_size[0] * sin_a + s * sin_a
    return (x, y, z)


def ramp_normal(incline_angle: float) -> tuple[float, float, float]:
    """Calculate the normal vector of the ramp surface."""
    return (math.sin(incline_angle), 0.0, math.cos(incline_angle))


def ramp_uphill_unit(incline_angle: float) -> tuple[float, float, float]:
    """Calculate the unit vector pointing uphill along the ramp."""
    return (-math.cos(incline_angle), 0.0, math.sin(incline_angle))


def ramp_downhill_unit(incline_angle: float) -> tuple[float, float, float]:
    """Calculate the unit vector pointing downhill along the ramp."""
    return (math.cos(incline_angle), 0.0, -math.sin(incline_angle))


def with_ids(configs: list[dict[str, Any]], level_id: int, start_id: int, seed: int, shuffle: bool = True) -> list[SampleSpec]:
    rng = np.random.default_rng(seed + level_id * 1000)
    order = np.arange(len(configs))
    if shuffle:
        rng.shuffle(order)
    samples = []
    for local_idx, cfg_idx in enumerate(order):
        cfg = configs[int(cfg_idx)]
        sample_id = start_id + local_idx
        samples.append(
            SampleSpec(
                level_id=level_id,
                sample_id=sample_id,
                level_name=cfg["level_name"],
                subtask=cfg["subtask"],
                main_variable=cfg["main_variable"],
                objects=cfg["objects"],
            )
        )
    return samples


def cycle_values(values: list[Any], count: int, rng: np.random.Generator) -> list[Any]:
    repeated = list(itertools.islice(itertools.cycle(values), count))
    rng.shuffle(repeated)
    return repeated


def balanced_counts(values: list[Any], count: int) -> dict[Any, int]:
    base = count // len(values)
    extra = count % len(values)
    return {value: base + (1 if idx < extra else 0) for idx, value in enumerate(values)}


def pick(values: list[Any], rng: np.random.Generator) -> Any:
    return values[int(rng.integers(0, len(values)))]


def quat_multiply(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def make_cfg(level_name: str, subtask: str, main_variable: str, objects: tuple[ObjectSpec, ...]) -> dict[str, Any]:
    return {
        "level_name": level_name,
        "subtask": subtask,
        "main_variable": main_variable,
        "objects": objects,
    }


def build_s3_stratified_level_configs(level_id: int, start_id: int, seed: int, count: int) -> list[SampleSpec]:
    rng = np.random.default_rng(seed + level_id * 1000)
    colors = ["red", "blue", "yellow", "green"]
    ramp_size = (2.4, 1.0, 0.12)
    configs: list[dict[str, Any]] = []

    if level_id == 1:
        # Level 1: 斜面倾角的影响 (theta -> 下滑加速度/静止临界)
        incline_angles = cycle_values([0.12, 0.22, 0.30, 0.38, 0.46, 0.56], count, rng)
        ramp_s_values = cycle_values([-0.65, -0.45, -0.25], count, rng)
        ramp_y_values = cycle_values([-0.2, 0.2], count, rng)
        color_values = cycle_values(colors, count, rng)
        for angle, ramp_s, ramp_y, color in zip(incline_angles, ramp_s_values, ramp_y_values, color_values):
            ramp, _ = ramp_spec(RAMP_OBJECT_ID, ramp_size, angle, lateral_friction=0.2, color_name="brown")
            surf_pos = ramp_surface_point(ramp_s, ramp_y, angle, ramp_size)
            normal = ramp_normal(angle)
            size = (0.24, 0.24, 0.24)
            pos = (surf_pos[0] + normal[0] * size[2] / 2, surf_pos[1] + normal[1] * size[2] / 2, surf_pos[2] + normal[2] * size[2] / 2)
            half_a = angle / 2.0
            quat = (math.cos(half_a), 0.0, math.sin(half_a), 0.0)
            obj = cube_spec(PRIMARY_OBJECT_ID, size, pos, (0.0, 0.0, 0.0), quaternion=quat, mass=1.0, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("incline_angle", "theta_to_slide_acceleration", "incline_angle",
                                   (ground_spec(), ramp, obj)))

    elif level_id == 2:
        # Level 2: 斜面摩擦系数的影响 (mu -> 临界角与加速度)
        friction_values = cycle_values([0.05, 0.1, 0.2, 0.28, 0.5, 0.7], count, rng)
        ramp_s_values = cycle_values([-0.65, -0.45, -0.25], count, rng)
        ramp_y_values = cycle_values([-0.2, 0.2], count, rng)
        color_values = cycle_values(colors, count, rng)
        angle = 0.35
        for friction, ramp_s, ramp_y, color in zip(friction_values, ramp_s_values, ramp_y_values, color_values):
            ramp, _ = ramp_spec(RAMP_OBJECT_ID, ramp_size, angle, lateral_friction=friction, color_name="brown")
            surf_pos = ramp_surface_point(ramp_s, ramp_y, angle, ramp_size)
            normal = ramp_normal(angle)
            size = (0.24, 0.24, 0.24)
            pos = (surf_pos[0] + normal[0] * size[2] / 2, surf_pos[1] + normal[1] * size[2] / 2, surf_pos[2] + normal[2] * size[2] / 2)
            half_a = angle / 2.0
            quat = (math.cos(half_a), 0.0, math.sin(half_a), 0.0)
            obj = cube_spec(PRIMARY_OBJECT_ID, size, pos, (0.0, 0.0, 0.0), quaternion=quat, mass=1.0, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("ramp_friction", "mu_to_critical_angle", "ramp.lateralFriction",
                                   (ground_spec(), ramp, obj)))

    elif level_id == 3:
        # Level 3: 初速度大小的影响 (v0 -> 滑行距离/停止)
        speed_values = cycle_values([1.2, 1.5, 1.8, 2.1, 2.4], count, rng)
        friction_values = cycle_values([0.15, 0.2], count, rng)
        color_values = cycle_values(colors, count, rng)
        angle = 0.35
        for speed, friction, color in zip(speed_values, friction_values, color_values):
            ramp, _ = ramp_spec(RAMP_OBJECT_ID, ramp_size, angle, lateral_friction=friction, color_name="brown")
            surf_pos = ramp_surface_point(0.15, 0.0, angle, ramp_size)
            normal = ramp_normal(angle)
            uphill = ramp_uphill_unit(angle)
            size = (0.24, 0.24, 0.24)
            pos = (surf_pos[0] + normal[0] * size[2] / 2, surf_pos[1] + normal[1] * size[2] / 2, surf_pos[2] + normal[2] * size[2] / 2)
            vel = (speed * uphill[0], speed * uphill[1], speed * uphill[2])
            half_a = angle / 2.0
            quat = (math.cos(half_a), 0.0, math.sin(half_a), 0.0)
            obj = cube_spec(PRIMARY_OBJECT_ID, size, pos, vel, quaternion=quat, mass=1.0, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("initial_velocity", "v0_to_slide_distance", "uphill_speed",
                                   (ground_spec(), ramp, obj)))

    elif level_id == 4:
        # Level 4: 质量无关性验证 (质量不影响加速度)
        mass_values = cycle_values([0.3, 0.5, 1.0, 2.0, 5.0], count, rng)
        friction_values = cycle_values([0.15, 0.2, 0.25], count, rng)
        color_values = cycle_values(colors, count, rng)
        angle = 0.35
        for mass, friction, color in zip(mass_values, friction_values, color_values):
            ramp, _ = ramp_spec(RAMP_OBJECT_ID, ramp_size, angle, lateral_friction=friction, color_name="brown")
            surf_pos = ramp_surface_point(-0.45, 0.0, angle, ramp_size)
            normal = ramp_normal(angle)
            size = (0.24, 0.24, 0.24)
            pos = (surf_pos[0] + normal[0] * size[2] / 2, surf_pos[1] + normal[1] * size[2] / 2, surf_pos[2] + normal[2] * size[2] / 2)
            half_a = angle / 2.0
            quat = (math.cos(half_a), 0.0, math.sin(half_a), 0.0)
            obj = cube_spec(PRIMARY_OBJECT_ID, size, pos, (0.0, 0.0, 0.0), quaternion=quat, mass=mass, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("mass_irrelevance", "mass_does_not_change_acceleration", "mass",
                                   (ground_spec(), ramp, obj)))

    elif level_id == 5:
        # Level 5: 尺寸/接触面积无关性 (面积不影响加速度)
        size_values = cycle_values([(0.18, 0.18, 0.18), (0.24, 0.24, 0.24), (0.30, 0.30, 0.30)], count, rng)
        speed_values = cycle_values([0.0, 0.6, 1.2], count, rng)
        friction_values = cycle_values([0.15, 0.2, 0.25], count, rng)
        color_values = cycle_values(colors, count, rng)
        angle = 0.35
        for size, speed, friction, color in zip(size_values, speed_values, friction_values, color_values):
            ramp, _ = ramp_spec(RAMP_OBJECT_ID, ramp_size, angle, lateral_friction=friction, color_name="brown")
            surf_pos = ramp_surface_point(-0.45, 0.0, angle, ramp_size)
            normal = ramp_normal(angle)
            downhill = ramp_downhill_unit(angle)
            pos = (surf_pos[0] + normal[0] * size[2] / 2, surf_pos[1] + normal[1] * size[2] / 2, surf_pos[2] + normal[2] * size[2] / 2)
            vel = (speed * downhill[0], speed * downhill[1], speed * downhill[2])
            half_a = angle / 2.0
            quat = (math.cos(half_a), 0.0, math.sin(half_a), 0.0)
            obj = cube_spec(PRIMARY_OBJECT_ID, size, pos, vel, quaternion=quat, mass=1.0, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("size_irrelevance", "size_does_not_change_acceleration", "size",
                                   (ground_spec(), ramp, obj)))

    elif level_id == 6:
        # Level 6: 初速度方向与斜面滑动 (向上/向下/零)
        angle_values = cycle_values([0.26, 0.35], count, rng)
        speed_values = cycle_values([0, 1.2, 1.5, 1.8, 2.2], count, rng)
        mode_values = cycle_values(["downhill", "uphill", "zero"], count, rng)
        friction_values = cycle_values([0.15, 0.2], count, rng)
        color_values = cycle_values(colors, count, rng)
        for angle, speed, mode, friction, color in zip(angle_values, speed_values, mode_values, friction_values, color_values):
            if speed == 0:
                mode = "zero"
            ramp, _ = ramp_spec(RAMP_OBJECT_ID, ramp_size, angle, lateral_friction=friction, color_name="brown")
            surf_pos = ramp_surface_point(-0.05, 0.0, angle, ramp_size)
            normal = ramp_normal(angle)
            uphill = ramp_uphill_unit(angle)
            downhill = ramp_downhill_unit(angle)
            size = (0.24, 0.24, 0.24)
            pos = (surf_pos[0] + normal[0] * size[2] / 2, surf_pos[1] + normal[1] * size[2] / 2, surf_pos[2] + normal[2] * size[2] / 2)
            if mode == "uphill":
                vel = (speed * uphill[0], speed * uphill[1], speed * uphill[2])
            elif mode == "downhill":
                vel = (speed * downhill[0], speed * downhill[1], speed * downhill[2])
            else:
                vel = (0.0, 0.0, 0.0)
            half_a = angle / 2.0
            quat = (math.cos(half_a), 0.0, math.sin(half_a), 0.0)
            obj = cube_spec(PRIMARY_OBJECT_ID, size, pos, vel, quaternion=quat, mass=1.0, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("velocity_direction", "direction_affects_slide", "velocity_mode",
                                   (ground_spec(), ramp, obj)))

    elif level_id == 7:
        # Level 7: 球体纯滑动到滚动的转换 (rollingFriction=0)
        angle_values = cycle_values([0.26, 0.35, 0.44], count, rng)
        radius_values = cycle_values([0.18, 0.22, 0.28], count, rng)
        mass_values = cycle_values([0.5, 1.0, 2.0], count, rng)
        friction_values = cycle_values([0.15, 0.2], count, rng)
        color_values = cycle_values(colors, count, rng)
        for angle, radius, mass, friction, color in zip(angle_values, radius_values, mass_values, friction_values, color_values):
            ramp, _ = ramp_spec(RAMP_OBJECT_ID, ramp_size, angle, lateral_friction=friction, color_name="brown")
            surf_pos = ramp_surface_point(-0.55, 0.0, angle, ramp_size)
            normal = ramp_normal(angle)
            pos = (surf_pos[0] + normal[0] * radius, surf_pos[1] + normal[1] * radius, surf_pos[2] + normal[2] * radius)
            obj = sphere_spec(PRIMARY_OBJECT_ID, radius, pos, (0.0, 0.0, 0.0), mass=mass, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("sphere_sliding_rolling", "sliding_to_rolling_on_ramp", "radius",
                                   (ground_spec(), ramp, obj)))

    elif level_id == 8:
        # Level 8: 滚动摩擦对球体斜面运动的影响
        angle_values = cycle_values([0.35, 0.42, 0.50], count, rng)
        radius_values = cycle_values([0.18, 0.22, 0.28], count, rng)
        rolling_values = cycle_values([0.0, 0.01, 0.03, 0.05, 0.1], count, rng)
        color_values = cycle_values(colors, count, rng)
        spinning_options = {0.0: [0.0], 0.01: [0.0, 0.003, 0.005], 0.03: [0.0, 0.003, 0.005], 0.05: [0.0, 0.003, 0.005], 0.1: [0.0, 0.003, 0.005]}
        for idx, (angle, radius, rolling, color) in enumerate(zip(angle_values, radius_values, rolling_values, color_values)):
            spinning = spinning_options[rolling][idx % len(spinning_options[rolling])]
            ramp, _ = ramp_spec(RAMP_OBJECT_ID, ramp_size, angle, lateral_friction=0.3, rolling_friction=rolling, spinning_friction=spinning, color_name="brown")
            surf_pos = ramp_surface_point(-0.55, 0.0, angle, ramp_size)
            normal = ramp_normal(angle)
            pos = (surf_pos[0] + normal[0] * radius, surf_pos[1] + normal[1] * radius, surf_pos[2] + normal[2] * radius)
            obj = sphere_spec(PRIMARY_OBJECT_ID, radius, pos, (0.0, 0.0, 0.0), mass=1.0, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("rolling_friction", "rolling_friction_affects_deceleration", "rollingFriction",
                                   (ground_spec(), ramp, obj)))

    elif level_id == 9:
        # Level 9: 圆柱斜面滑动行为——姿态与方向
        angle_values = cycle_values([0.26, 0.35], count, rng)
        radius_values = cycle_values([0.16, 0.20, 0.24], count, rng)
        height_values = cycle_values([0.22, 0.28, 0.34], count, rng)
        speed_values = cycle_values([0.0, 0.6, 1.0], count, rng)
        mode_values = cycle_values(["downhill", "uphill", "zero"], count, rng)
        friction_values = cycle_values([0.15, 0.2], count, rng)
        rolling_values = cycle_values([0.0, 0.01, 0.05], count, rng)
        color_values = cycle_values(colors, count, rng)
        orientation_idx = 0
        for angle, radius, height, speed, mode, friction, rolling, color in zip(
            angle_values, radius_values, height_values, speed_values, mode_values, friction_values, rolling_values, color_values
        ):
            if speed == 0:
                mode = "zero"
            ramp, _ = ramp_spec(RAMP_OBJECT_ID, ramp_size, angle, lateral_friction=friction, rolling_friction=rolling, color_name="brown")
            orientation = orientation_idx % 3
            orientation_idx += 1
            uphill = ramp_uphill_unit(angle)
            downhill = ramp_downhill_unit(angle)
            normal = ramp_normal(angle)
            half_a = angle / 2.0
            ramp_quat = (math.cos(half_a), 0.0, math.sin(half_a), 0.0)
            if orientation == 0:  # upright
                surf_pos = ramp_surface_point(-0.45, 0.0, angle, ramp_size)
                pos = (surf_pos[0] + normal[0] * height / 2, surf_pos[1] + normal[1] * height / 2, surf_pos[2] + normal[2] * height / 2)
                quat = quat_multiply(ramp_quat, (1.0, 0.0, 0.0, 0.0))
            elif orientation == 1:  # lying_axis_x
                surf_pos = ramp_surface_point(-0.45, 0.0, angle, ramp_size)
                pos = (surf_pos[0] + normal[0] * radius, surf_pos[1] + normal[1] * radius, surf_pos[2] + normal[2] * radius)
                quat = quat_multiply(ramp_quat, (0.7071, 0.0, 0.7071, 0.0))
            else:  # lying_axis_y
                surf_pos = ramp_surface_point(-0.45, 0.0, angle, ramp_size)
                pos = (surf_pos[0] + normal[0] * radius, surf_pos[1] + normal[1] * radius, surf_pos[2] + normal[2] * radius)
                quat = quat_multiply(ramp_quat, (0.7071, 0.7071, 0.0, 0.0))
            if mode == "uphill":
                vel = (speed * uphill[0], speed * uphill[1], speed * uphill[2])
            elif mode == "downhill":
                vel = (speed * downhill[0], speed * downhill[1], speed * downhill[2])
            else:
                vel = (0.0, 0.0, 0.0)
            obj = cylinder_spec(PRIMARY_OBJECT_ID, radius, height, pos, vel, quaternion=quat, mass=1.0,
                               lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("cylinder_sliding", "orientation_affects_sliding", "initial_quaternion",
                                   (ground_spec(), ramp, obj)))

    else:
        raise ValueError(f"S3 supports levels 1..9, got L{level_id}")

    for idx, cfg in enumerate(configs):
        ramp_color = "brown" if idx % 2 == 0 else "gray"
        cfg["objects"] = tuple(
            replace(obj, color_name=ramp_color) if obj.object_type == "ramp" else obj
            for obj in cfg["objects"]
        )

    return with_ids(configs, level_id, start_id, seed, shuffle=False)


def target_count_for_level(level_id: int, requested_count: int | None) -> int:
    if requested_count is not None:
        return requested_count
    if level_id not in LEVEL_TARGETS:
        raise ValueError(f"S3 supports levels 1..9, got L{level_id}")
    return LEVEL_TARGETS[level_id]


def take_samples(level_id: int, start_id: int, seed: int, count: int | None) -> list[SampleSpec]:
    target_count = target_count_for_level(level_id, count)
    return build_s3_stratified_level_configs(level_id, start_id, seed, target_count)


def make_camera(view_name: str, spec: dict[str, Any]):
    if spec["type"] == "Perspective":
        return kb.PerspectiveCamera(
            name=f"camera_{view_name}",
            position=spec["position"],
            look_at=spec["look_at"],
            focal_length=spec["focal_length"],
            sensor_width=spec["sensor_width"],
        )
    if spec["type"] == "Orthographic":
        return kb.OrthographicCamera(
            name=f"camera_{view_name}",
            position=spec["position"],
            look_at=spec["look_at"],
            orthographic_scale=spec["orthographic_scale"],
        )
    raise ValueError(f"Unsupported camera type: {spec['type']}")


def build_asset(spec: ObjectSpec):
    physics_mass = 0.0 if spec.static else spec.mass
    if physics_mass is None:
        raise ValueError(f"Dynamic object {spec.object_id} is missing mass")
    kwargs = {
        "name": spec.name,
        "position": spec.position,
        "quaternion": spec.quaternion,
        "velocity": spec.velocity,
        "angular_velocity": spec.angular_velocity,
        "static": spec.static,
        "mass": physics_mass,
        "friction": spec.lateral_friction,
        "restitution": spec.restitution,
        "material": material(spec.color_name),
        "segmentation_id": spec.object_id,
    }
    if spec.object_type == "sphere":
        assert spec.radius is not None
        return kb.Sphere(scale=spec.radius, **kwargs)
    if spec.object_type in {"cube", "ground", "ramp", "wall"}:
        assert spec.size is not None
        scale = tuple(v / 2.0 for v in spec.size)
        return kb.Cube(scale=scale, **kwargs)
    if spec.object_type == "cylinder":
        assert spec.radius is not None and spec.height is not None
        return kb.Cylinder(scale=(spec.radius, spec.radius, spec.height / 2.0), **kwargs)
    raise ValueError(f"Unsupported object type for S3: {spec.object_type}")


def build_scene(sample: SampleSpec, resolution: int, view_names: list[str]):
    scene = kb.Scene(resolution=(resolution, resolution))
    scene.frame_start = 0
    scene.frame_end = NUM_FRAMES - 1
    scene.frame_rate = FPS
    scene.step_rate = SIM_HZ
    scene.gravity = GRAVITY
    scene.background = (0.03, 0.035, 0.04, 1.0)
    scene.ambient_illumination = (0.08, 0.08, 0.08, 1.0)

    cameras = {name: make_camera(name, VIEWS[name]) for name in view_names}
    scene += list(cameras.values())
    scene.camera = cameras[view_names[0]]
    scene += kb.DirectionalLight(
        name="sun",
        position=(-3.0, -4.0, 7.0),
        look_at=(0.0, 0.0, 0.5),
        intensity=2.6,
    )

    assets_by_id = {}
    for spec in sample.objects:
        asset = build_asset(spec)
        scene += asset
        assets_by_id[spec.object_id] = asset

    return scene, cameras, assets_by_id


def apply_extra_dynamics(simulator: PyBullet, sample: SampleSpec, assets_by_id: dict[int, Any]) -> None:
    client = simulator._physics_client
    for spec in sample.objects:
        asset = assets_by_id[spec.object_id]
        body_id = asset.linked_objects.get(simulator)
        if body_id is None:
            continue
        physics_mass = 0.0 if spec.static else spec.mass
        if physics_mass is None:
            raise ValueError(f"Dynamic object {spec.object_id} is missing mass")
        client.changeDynamics(
            body_id,
            -1,
            lateralFriction=spec.lateral_friction,
            rollingFriction=spec.rolling_friction,
            spinningFriction=spec.spinning_friction,
            restitution=spec.restitution,
            mass=physics_mass,
        )


def zero_force_matrix(object_ids: list[int]) -> list[list[Any]]:
    matrix: list[list[Any]] = [[None for _ in range(10)] for _ in range(10)]
    for i in object_ids:
        if i >= 10:
            continue
        for j in object_ids:
            if j < 10:
                matrix[i][j] = [0.0, 0.0, 0.0]
    return matrix


def add_force(matrix: list[list[Any]], row: int, col: int, vec: np.ndarray) -> None:
    if row >= 10 or col >= 10 or matrix[row][col] is None:
        return
    current = np.array(matrix[row][col], dtype=np.float64)
    matrix[row][col] = (current + vec).astype(float).tolist()


def average_force_matrix(matrix: list[list[Any]], divisor: int) -> list[list[Any]]:
    averaged: list[list[Any]] = []
    for row in matrix:
        averaged_row = []
        for cell in row:
            if cell is None:
                averaged_row.append(None)
            else:
                averaged_row.append((np.array(cell, dtype=np.float64) / divisor).astype(float).tolist())
        averaged.append(averaged_row)
    return averaged


def simulate_and_keyframe(
    scene,
    simulator: PyBullet,
    sample: SampleSpec,
    assets_by_id: dict[int, Any],
) -> tuple[list[dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    apply_extra_dynamics(simulator, sample, assets_by_id)
    object_ids = sorted(assets_by_id)
    body_to_object = {
        asset.linked_objects[simulator]: object_id
        for object_id, asset in assets_by_id.items()
        if simulator in asset.linked_objects
    }

    steps_per_frame = scene.step_rate // scene.frame_rate
    total_steps = NUM_FRAMES * steps_per_frame
    frame_states: list[dict[int, dict[str, Any]]] = []
    frame_forces = [zero_force_matrix(object_ids) for _ in range(NUM_FRAMES)]

    for step in range(total_steps):
        frame_idx = min(step // steps_per_frame, NUM_FRAMES - 1)
        contacts = simulator._physics_client.getContactPoints()
        for contact in contacts:
            body_a = int(contact[1])
            body_b = int(contact[2])
            normal_on_b = np.array(contact[7], dtype=np.float64)
            normal_force = float(contact[9])
            if normal_force <= 1e-6:
                continue
            object_a = body_to_object.get(body_a)
            object_b = body_to_object.get(body_b)
            if object_a is None or object_b is None:
                continue

            force_on_b_by_a = -normal_force * normal_on_b
            add_force(frame_forces[frame_idx], object_b, object_a, force_on_b_by_a)
            add_force(frame_forces[frame_idx], object_a, object_b, -force_on_b_by_a)

        if step % steps_per_frame == 0:
            states = {}
            for object_id, asset in assets_by_id.items():
                body_id = asset.linked_objects[simulator]
                position, quaternion = simulator.get_position_and_rotation(body_id)
                velocity, angular_velocity = simulator.get_velocities(body_id)
                states[object_id] = {
                    "position": [float(v) for v in position],
                    "quaternion": [float(v) for v in quaternion],
                    "velocity": [float(v) for v in velocity],
                    "angular_velocity": [float(v) for v in angular_velocity],
                }
            frame_states.append(states)

        simulator._physics_client.stepSimulation()

    for frame_idx, states in enumerate(frame_states):
        for object_id, state in states.items():
            asset = assets_by_id[object_id]
            asset.position = tuple(state["position"])
            asset.quaternion = tuple(state["quaternion"])
            asset.velocity = tuple(state["velocity"])
            asset.angular_velocity = tuple(state["angular_velocity"])
            asset.keyframe_insert("position", frame_idx)
            asset.keyframe_insert("quaternion", frame_idx)
            asset.keyframe_insert("velocity", frame_idx)
            asset.keyframe_insert("angular_velocity", frame_idx)

    averaged_forces = [average_force_matrix(matrix, steps_per_frame) for matrix in frame_forces]
    force_payloads = [
        {"object_order": object_ids, "force_matrix": averaged_forces[i]}
        for i in range(NUM_FRAMES)
    ]
    return frame_states, force_payloads


def make_dirs(sample_dir: Path) -> None:
    for frame_num in range(1, NUM_FRAMES + 1):
        frame_dir = sample_dir / "dynamic" / str(frame_num)
        (frame_dir / "object_dynamicjson").mkdir(parents=True, exist_ok=True)
        (frame_dir / "object_segment").mkdir(parents=True, exist_ok=True)


def static_json(sample: SampleSpec) -> list[dict[str, Any]]:
    rows = []
    for spec in sample.objects:
        rows.append(
            {
                "object_id": spec.object_id,
                "segmentation_id": spec.object_id,
                "object_type": spec.object_type,
                "static": spec.static,
                "radius": spec.radius,
                "size": list(spec.size) if spec.size is not None else None,
                "height": spec.height,
                "mass": spec.mass,
                "lateralFriction": spec.lateral_friction,
                "rollingFriction": spec.rolling_friction,
                "spinningFriction": spec.spinning_friction,
                "restitution": spec.restitution,
                "color_name": spec.color_name,
                "rgba": list(COLORS[spec.color_name]),
            }
        )
    return rows


def resultant_force(spec: ObjectSpec, force_matrix: list[list[Any]]) -> list[float]:
    total = np.zeros(3, dtype=np.float64)
    if spec.object_id < len(force_matrix):
        for force in force_matrix[spec.object_id]:
            if force is not None:
                total += np.array(force, dtype=np.float64)
    if not spec.static and spec.mass is not None:
        total += np.array(GRAVITY, dtype=np.float64) * spec.mass
    return total.astype(float).tolist()


def video_json(sample: SampleSpec, resolution: int, view_name: str) -> dict[str, Any]:
    spec = VIEWS[view_name]
    camera = {
        "view_name": view_name,
        "type": spec["type"],
        "position": list(spec["position"]),
        "look_at": list(spec["look_at"]),
        "resolution": [resolution, resolution],
        "video_path": f"{sample.sample_id}.mp4",
        "depth_path": f"{sample.sample_id}.npz",
    }
    if spec["type"] == "Perspective":
        camera["focal_length"] = spec["focal_length"]
        camera["sensor_width"] = spec["sensor_width"]
    else:
        camera["orthographic_scale"] = spec["orthographic_scale"]
    return {
        "duration_s": DURATION_S,
        "fps": FPS,
        "num_frames": NUM_FRAMES,
        "gravity": list(GRAVITY),
        "units": {
            "length": "m",
            "mass": "kg",
            "time": "s",
            "linear_velocity": "m/s",
            "angular_velocity": "rad/s",
            "angle": "rad",
            "force": "kg*m/s^2",
        },
        "cameras": [camera],
        "physics_hz": SIM_HZ,
        "level_name": sample.level_name,
        "subtask": sample.subtask,
        "main_variable": sample.main_variable,
    }


def rgba_to_mp4(rgba: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = rgba[..., :3]
    imageio.mimwrite(path, list(rgb), fps=fps, quality=8, macro_block_size=1)


def squeeze_layer(layer: np.ndarray) -> np.ndarray:
    if layer.ndim == 4 and layer.shape[-1] == 1:
        return layer[..., 0]
    return layer


def render_view(
    scene,
    renderer: Blender,
    camera: Any,
    sample_dir: Path,
    scratch_root: Path,
    sample_id: int,
    view_name: str,
) -> dict[str, np.ndarray]:
    scene.camera = camera
    renderer.scratch_dir = scratch_root / f"blender_{view_name}"
    frames = renderer.render(return_layers=("rgba", "depth", "segmentation"))
    frames["depth"] = squeeze_layer(frames["depth"]).astype(np.float32)
    frames["segmentation"] = squeeze_layer(frames["segmentation"]).astype(np.int32)
    rgba_to_mp4(frames["rgba"], sample_dir / f"{sample_id}.mp4", FPS)
    np.savez_compressed(
        sample_dir / f"{sample_id}.npz",
        depth=frames["depth"].astype(np.float32),
        unit=np.array("m"),
    )
    return frames


def replay_scene_keyframes(scene) -> None:
    # A renderer created after simulation must receive the keyframes already stored on assets.
    for asset in scene.assets:
        keyframes = getattr(asset, "keyframes", None)
        if not keyframes:
            continue
        for member, frame_values in list(keyframes.items()):
            original = getattr(asset, member)
            try:
                for frame, value in sorted(list(frame_values.items())):
                    setattr(asset, member, value)
                    asset.keyframe_insert(member, frame)
            finally:
                setattr(asset, member, original)


def unlink_renderer(scene, renderer: Blender) -> None:
    for asset in scene.assets:
        asset.unobserve_all()
        material_asset = getattr(asset, "material", None)
        if hasattr(material_asset, "unobserve_all"):
            material_asset.unobserve_all()
    for trait_name, setters in renderer.scene_observers.items():
        for setter in setters:
            scene.unobserve(setter, trait_name)
    scene.unlink_view(renderer)
    for asset in scene.assets:
        asset.linked_objects.pop(renderer, None)


def write_dynamic_outputs(
    sample_dir: Path,
    sample: SampleSpec,
    frame_states: list[dict[int, dict[str, Any]]],
    force_payloads: list[dict[str, Any]],
    rendered: dict[str, np.ndarray],
) -> None:
    write_json(sample_dir / "physics_labels.json", compute_physics_labels(sample, frame_states, force_payloads, FPS))
    object_ids = [spec.object_id for spec in sample.objects]
    specs_by_id = {spec.object_id: spec for spec in sample.objects}

    for frame_idx in range(NUM_FRAMES):
        frame_num = frame_idx + 1
        frame_dir = sample_dir / "dynamic" / str(frame_num)

        imageio.imwrite(frame_dir / f"{frame_num}.png", rendered["rgba"][frame_idx, ..., :3])

        write_json(frame_dir / "force_matrix.json", force_payloads[frame_idx])

        for object_id in object_ids:
            segmentation_id = object_id
            mask = rendered["segmentation"][frame_idx] == segmentation_id
            visible_area = int(mask.sum())
            np.savez_compressed(
                frame_dir / "object_segment" / f"{object_id}.npz",
                mask=mask.astype(np.uint8),
                object_id=np.array(object_id, dtype=np.int32),
                segmentation_id=np.array(segmentation_id, dtype=np.int32),
                frame_num=np.array(frame_num, dtype=np.int32),
            )

            state = frame_states[frame_idx][object_id]
            dynamic = {
                "object_id": object_id,
                "time": frame_idx / FPS,
                "visible_area": visible_area,
                "position": state["position"],
                "quaternion": state["quaternion"],
                "velocity": state["velocity"],
                "angular_velocity": state["angular_velocity"],
                "resultant force": resultant_force(specs_by_id[object_id], force_payloads[frame_idx]["force_matrix"]),
                "segmentation_path": f"../object_segment/{object_id}.npz",
            }
            write_json(frame_dir / "object_dynamicjson" / f"{object_id}.json", dynamic)


def sample_dir_for(output_root: str, level_id: int, sample_id: int) -> Path:
    return Path(output_root) / f"S{SCENE_ID}" / f"L{level_id}" / str(sample_id)


def generate_physical_sample(args: argparse.Namespace, sample: SampleSpec, output_start_id: int) -> None:
    load_runtime_dependencies()

    view_outputs = []
    for view_idx, view_name in enumerate(args.views):
        output_sample = replace(sample, sample_id=output_start_id + view_idx)
        sample_dir = sample_dir_for(args.output_root, output_sample.level_id, output_sample.sample_id)
        view_outputs.append((view_name, output_sample, sample_dir))

    for _, _, sample_dir in view_outputs:
        if sample_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"{sample_dir} already exists; pass --overwrite to replace it")
            shutil.rmtree(sample_dir)

    scratch_root = view_outputs[0][2] / "_scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    scene, cameras, assets_by_id = build_scene(sample, args.resolution, args.views)
    simulator = PyBullet(scene, scratch_root / "pybullet")

    frame_states, force_payloads = simulate_and_keyframe(scene, simulator, sample, assets_by_id)

    for view_name, output_sample, sample_dir in view_outputs:
        try:
            make_dirs(sample_dir)
            renderer = Blender(
                scene,
                scratch_root / f"blender_{view_name}",
                adaptive_sampling=True,
                use_denoising=True,
                samples_per_pixel=args.samples_per_pixel,
            )
            try:
                replay_scene_keyframes(scene)
                rendered = render_view(
                    scene,
                    renderer,
                    cameras[view_name],
                    sample_dir,
                    scratch_root,
                    output_sample.sample_id,
                    view_name,
                )
            finally:
                unlink_renderer(scene, renderer)
                gc.collect()
            write_json(sample_dir / "video.json", video_json(output_sample, args.resolution, view_name))
            write_json(sample_dir / "object_static.json", static_json(output_sample))
            write_dynamic_outputs(sample_dir, output_sample, frame_states, force_payloads, rendered)
        except Exception:
            shutil.rmtree(sample_dir, ignore_errors=True)
            raise

    if not args.keep_scratch:
        shutil.rmtree(scratch_root, ignore_errors=True)


def main() -> None:
    args = parse_args()
    if len(args.views) != len(set(args.views)):
        raise ValueError("--views cannot contain duplicates")
    for level_id in args.levels:
        physical_samples = take_samples(level_id, 1, args.seed, args.samples_per_level)
        output_id = args.start_id
        for physical_idx, sample in enumerate(physical_samples, start=1):
            last_output_id = output_id + len(args.views) - 1
            print(
                f"Generating S{SCENE_ID}/L{sample.level_id} physical {physical_idx} "
                f"as video dirs {output_id}-{last_output_id}: {sample.level_name}"
            )
            generate_physical_sample(args, sample, output_id)
            output_id += len(args.views)


if __name__ == "__main__":
    main()
