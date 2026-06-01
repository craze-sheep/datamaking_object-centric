#!/usr/bin/env python3
"""Generate S4 horizontal sliding samples with Kubric.

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
SCENE_ID = 4
GROUND_OBJECT_ID = 1
PRIMARY_OBJECT_ID = 2
WALL_OBJECT_ID = 3
LEVEL_TARGETS = {1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 150, 7: 150, 8: 150, 9: 150}
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
    parser.add_argument("--output_root", default="database1")
    parser.add_argument("--levels", nargs="+", type=int, default=[1])
    parser.add_argument("--samples_per_level", type=int, default=None)
    parser.add_argument("--start_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--views", nargs="+", choices=sorted(VIEWS), default=["front", "top"])
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


def ground_spec(restitution: float = 0.0, friction: float = 0.0) -> ObjectSpec:
    size = (10.0, 5.0, 0.08)
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



def wall_spec(
    object_id: int,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    restitution: float = 0.5,
    lateral_friction: float = 0.0,
    color_name: str = "gray",
) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        object_type="wall",
        name="wall",
        static=True,
        size=size,
        position=position,
        lateral_friction=lateral_friction,
        restitution=restitution,
        color_name=color_name,
    )


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


def make_cfg(level_name: str, subtask: str, main_variable: str, objects: tuple[ObjectSpec, ...]) -> dict[str, Any]:
    return {
        "level_name": level_name,
        "subtask": subtask,
        "main_variable": main_variable,
        "objects": objects,
    }


def build_s4_stratified_level_configs(level_id: int, start_id: int, seed: int, count: int) -> list[SampleSpec]:
    rng = np.random.default_rng(seed + level_id * 1000)
    colors = ["red", "blue", "yellow", "green"]
    wall_size = (0.10, 5.00, 1.20)
    wall_pos = (1.20, 0.0, 0.60)
    configs: list[dict[str, Any]] = []

    if level_id == 1:
        # Level 1: 墙面恢复系数的影响 (restitution -> 反弹速度比)
        restitution_values = cycle_values([0.0, 0.3, 0.5, 0.8, 1.0], count, rng)
        x_positions = cycle_values([-0.75, -0.55, -0.35], count, rng)
        y_positions = cycle_values([-0.25, 0.0, 0.25], count, rng)
        color_values = cycle_values(colors, count, rng)
        for restitution, x, y, color in zip(restitution_values, x_positions, y_positions, color_values):
            wall = wall_spec(WALL_OBJECT_ID, wall_size, wall_pos, restitution=restitution, lateral_friction=0.0, color_name="gray")
            obj = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (x, y, 0.22), (2.0, 0.0, 0.0), mass=1.0, restitution=1.0, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("wall_restitution", "restitution_to_rebound_speed", "wall.restitution",
                                   (ground_spec(), wall, obj)))

    elif level_id == 2:
        # Level 2: 入射速度大小的影响 (v0 -> 反弹后距离)
        speed_values = cycle_values([1.4, 1.7, 2.0, 2.4, 2.8], count, rng)
        x_positions = cycle_values([-0.75, -0.55, -0.35], count, rng)
        y_positions = cycle_values([-0.25, 0.0, 0.25], count, rng)
        color_values = cycle_values(colors, count, rng)
        for speed, x, y, color in zip(speed_values, x_positions, y_positions, color_values):
            wall = wall_spec(WALL_OBJECT_ID, wall_size, wall_pos, restitution=0.5, lateral_friction=0.0, color_name="gray")
            obj = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (x, y, 0.22), (speed, 0.0, 0.0), mass=1.0, restitution=1.0, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("initial_velocity", "v0_to_rebound_distance", "speed",
                                   (ground_spec(), wall, obj)))

    elif level_id == 3:
        # Level 3: 质量无关性验证 (质量不影响理想碰撞恢复规律)
        mass_values = cycle_values([0.3, 0.5, 1.0, 2.0, 5.0], count, rng)
        speed_values = cycle_values([1.6, 2.2], count, rng)
        restitution_values = cycle_values([0.5, 0.8], count, rng)
        y_positions = cycle_values([-0.25, 0.0, 0.25], count, rng)
        color_values = cycle_values(colors, count, rng)
        for mass, speed, restitution, y, color in zip(mass_values, speed_values, restitution_values, y_positions, color_values):
            wall = wall_spec(WALL_OBJECT_ID, wall_size, wall_pos, restitution=restitution, lateral_friction=0.0, color_name="gray")
            obj = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (-0.65, y, 0.22), (speed, 0.0, 0.0), mass=mass, restitution=1.0, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("mass_irrelevance", "mass_does_not_change_restitution", "mass",
                                   (ground_spec(), wall, obj)))

    elif level_id == 4:
        # Level 4: 物体尺寸无关性验证 (球半径/方块尺寸不影响反弹速度比)
        object_type_idx = cycle_values([0, 1], count, rng)  # 0=sphere, 1=cube
        speed_values = cycle_values([1.6, 2.2], count, rng)
        restitution_values = cycle_values([0.5, 0.8], count, rng)
        y_positions = cycle_values([-0.2, 0.2], count, rng)
        color_values = cycle_values(colors, count, rng)
        size_values_sphere = cycle_values([0.18, 0.22, 0.28], count, rng)
        size_values_cube = cycle_values([(0.18, 0.18, 0.18), (0.24, 0.24, 0.24), (0.30, 0.30, 0.30)], count, rng)
        for i, (speed, restitution, y, color) in enumerate(zip(speed_values, restitution_values, y_positions, color_values)):
            wall = wall_spec(WALL_OBJECT_ID, wall_size, wall_pos, restitution=restitution, lateral_friction=0.0, color_name="gray")
            if object_type_idx[i] == 0:
                radius = size_values_sphere[i]
                obj = sphere_spec(PRIMARY_OBJECT_ID, radius, (-0.65, y, radius), (speed, 0.0, 0.0), mass=1.0, restitution=1.0, lateral_friction=0.0, color_name=color)
            else:
                size = size_values_cube[i]
                obj = cube_spec(PRIMARY_OBJECT_ID, size, (-0.65, y, size[2] / 2), (speed, 0.0, 0.0), mass=1.0, restitution=1.0, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("size_irrelevance", "size_does_not_change_restitution", "size",
                                   (ground_spec(), wall, obj)))

    elif level_id == 5:
        # Level 5: 物体恢复系数的影响 (物体 restitution 与墙面 restitution 组合)
        wall_restitution_values = cycle_values([0.3, 0.5, 1.0], count, rng)
        obj_restitution_values = cycle_values([0.3, 0.5, 1.0], count, rng)
        speed_values = cycle_values([1.6, 2.2], count, rng)
        x_positions = cycle_values([-0.75, -0.55], count, rng)
        y_positions = cycle_values([-0.25, 0.0, 0.25], count, rng)
        color_values = cycle_values(colors, count, rng)
        for wall_e, obj_e, speed, x, y, color in zip(wall_restitution_values, obj_restitution_values, speed_values, x_positions, y_positions, color_values):
            wall = wall_spec(WALL_OBJECT_ID, wall_size, wall_pos, restitution=wall_e, lateral_friction=0.0, color_name="gray")
            obj = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (x, y, 0.22), (speed, 0.0, 0.0), mass=1.0, restitution=obj_e, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("combined_restitution", "combined_restitution_effect", "object.restitution",
                                   (ground_spec(), wall, obj)))

    elif level_id == 6:
        # Level 6: 入射角度的影响 (斜碰，法向速度与切向速度分解)
        speed_values = cycle_values([1.8, 2.4], count, rng)
        angle_values = cycle_values([-45, -30, -15, 0, 15, 30, 45], count, rng)
        restitution_values = cycle_values([0.5, 0.8], count, rng)
        x_positions = cycle_values([-0.80, -0.60], count, rng)
        y_positions = cycle_values([-0.35, 0.0, 0.35], count, rng)
        color_values = cycle_values(colors, count, rng)
        for speed, angle_deg, restitution, x, y, color in zip(speed_values, angle_values, restitution_values, x_positions, y_positions, color_values):
            angle_rad = math.radians(angle_deg)
            vx = speed * math.cos(angle_rad)
            vy = speed * math.sin(angle_rad)
            wall = wall_spec(WALL_OBJECT_ID, wall_size, wall_pos, restitution=restitution, lateral_friction=0.0, color_name="gray")
            obj = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (x, y, 0.22), (vx, vy, 0.0), mass=1.0, restitution=1.0, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("incident_angle", "angle_affects_rebound", "incident_angle_degree",
                                   (ground_spec(), wall, obj)))

    elif level_id == 7:
        # Level 7: 墙面摩擦系数的影响 (斜碰时切向速度损失)
        friction_values = cycle_values([0.0, 0.2, 0.5, 0.8, 1.0], count, rng)
        speed_values = cycle_values([2.0, 2.6], count, rng)
        angle_values = cycle_values([-45, -30, 30, 45], count, rng)
        x_positions = cycle_values([-0.80, -0.60], count, rng)
        y_positions = cycle_values([-0.35, 0.0, 0.35], count, rng)
        color_values = cycle_values(colors, count, rng)
        for friction, speed, angle_deg, x, y, color in zip(friction_values, speed_values, angle_values, x_positions, y_positions, color_values):
            angle_rad = math.radians(angle_deg)
            vx = speed * math.cos(angle_rad)
            vy = speed * math.sin(angle_rad)
            wall = wall_spec(WALL_OBJECT_ID, wall_size, wall_pos, restitution=0.8, lateral_friction=friction, color_name="gray")
            obj = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (x, y, 0.22), (vx, vy, 0.0), mass=1.0, restitution=1.0, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("wall_friction", "friction_affects_tangential_velocity", "wall.lateralFriction",
                                   (ground_spec(), wall, obj)))

    elif level_id == 8:
        # Level 8: 球体碰撞与旋转效应 (入射角速度影响反弹轨迹)
        radius_values = cycle_values([0.18, 0.22, 0.28], count, rng)
        velocity_values = cycle_values([(2.2, 0, 0), (2.2, 0.6, 0), (2.2, -0.6, 0)], count, rng)
        spin_modes = cycle_values(["no_spin", "spin_z_pos", "spin_z_neg", "spin_y_pos", "spin_y_neg"], count, rng)
        friction_values = cycle_values([0.6, 1.0], count, rng)
        x_positions = cycle_values([-0.75, -0.55], count, rng)
        y_positions = cycle_values([-0.25, 0.25], count, rng)
        color_values = cycle_values(colors, count, rng)
        for radius, vel, spin, friction, x, y, color in zip(radius_values, velocity_values, spin_modes, friction_values, x_positions, y_positions, color_values):
            if spin == "no_spin":
                ang_vel = (0, 0, 0)
            elif spin == "spin_z_pos":
                ang_vel = (0, 0, 8)
            elif spin == "spin_z_neg":
                ang_vel = (0, 0, -8)
            elif spin == "spin_y_pos":
                ang_vel = (0, 8, 0)
            else:
                ang_vel = (0, -8, 0)
            wall = wall_spec(WALL_OBJECT_ID, wall_size, wall_pos, restitution=0.8, lateral_friction=friction, color_name="gray")
            obj = sphere_spec(PRIMARY_OBJECT_ID, radius, (x, y, radius), vel, angular_velocity=ang_vel, mass=1.0, restitution=1.0, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("spin_effect", "spin_affects_rebound", "initial_angular_velocity",
                                   (ground_spec(), wall, obj)))

    elif level_id == 9:
        # Level 9: 形状泛化 (sphere vs cube，正碰与斜碰)
        object_type_idx = cycle_values([0, 1], count, rng)  # 0=sphere, 1=cube
        speed_values = cycle_values([1.8, 2.4], count, rng)
        angle_values = cycle_values([0, -30, 30], count, rng)
        restitution_values = cycle_values([0.5, 0.8], count, rng)
        friction_values = cycle_values([0.0, 0.5], count, rng)
        y_positions = cycle_values([-0.25, 0.25], count, rng)
        radius_values = cycle_values([0.18, 0.22, 0.28], count, rng)
        size_values = cycle_values([(0.18, 0.18, 0.18), (0.24, 0.24, 0.24), (0.30, 0.30, 0.30)], count, rng)
        color_values = cycle_values(colors, count, rng)
        for i, (speed, angle_deg, restitution, friction, y, color) in enumerate(zip(speed_values, angle_values, restitution_values, friction_values, y_positions, color_values)):
            angle_rad = math.radians(angle_deg)
            vx = speed * math.cos(angle_rad)
            vy = speed * math.sin(angle_rad)
            wall = wall_spec(WALL_OBJECT_ID, wall_size, wall_pos, restitution=restitution, lateral_friction=friction, color_name="gray")
            if object_type_idx[i] == 0:
                radius = radius_values[i]
                obj = sphere_spec(PRIMARY_OBJECT_ID, radius, (-0.70, y, radius), (vx, vy, 0.0), mass=1.0, restitution=1.0, lateral_friction=1.0, color_name=color)
            else:
                # cube with optional yaw rotation
                yaw = 0.3827 if angle_deg != 0 else 0.0
                quat = (0.9239, 0, 0, yaw) if yaw > 0 else (1.0, 0.0, 0.0, 0.0)
                size = size_values[i]
                obj = cube_spec(PRIMARY_OBJECT_ID, size, (-0.70, y, size[2] / 2), (vx, vy, 0.0), quaternion=quat, mass=1.0, restitution=1.0, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("shape_generalization", "sphere_vs_cube_bounce", "object_type",
                                   (ground_spec(), wall, obj)))

    else:
        raise ValueError(f"S4 supports levels 1..9, got L{level_id}")

    for idx, cfg in enumerate(configs):
        wall_color = "gray" if idx % 2 == 0 else "blue"
        cfg["objects"] = tuple(
            replace(obj, color_name=wall_color) if obj.object_type == "wall" else obj
            for obj in cfg["objects"]
        )

    return with_ids(configs, level_id, start_id, seed, shuffle=False)


def target_count_for_level(level_id: int, requested_count: int | None) -> int:
    if requested_count is not None:
        return requested_count
    if level_id not in LEVEL_TARGETS:
        raise ValueError(f"S4 supports levels 1..9, got L{level_id}")
    return LEVEL_TARGETS[level_id]


def take_samples(level_id: int, start_id: int, seed: int, count: int | None) -> list[SampleSpec]:
    target_count = target_count_for_level(level_id, count)
    return build_s4_stratified_level_configs(level_id, start_id, seed, target_count)


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
    if spec.object_type in {"cube", "ground", "wall"}:
        assert spec.size is not None
        scale = tuple(v / 2.0 for v in spec.size)
        return kb.Cube(scale=scale, **kwargs)
    if spec.object_type == "cylinder":
        assert spec.radius is not None and spec.height is not None
        return kb.Cylinder(scale=(spec.radius, spec.radius, spec.height / 2.0), **kwargs)
    raise ValueError(f"Unsupported object type for S4: {spec.object_type}")


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
