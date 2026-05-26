#!/usr/bin/env python3
"""Generate S7 horizontal sliding samples with Kubric.

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
SCENE_ID = 7
GROUND_OBJECT_ID = 1
PRIMARY_OBJECT_ID = 2
RELAY_OBJECT_ID = 3
TERMINAL_OBJECT_ID = 4
LEVEL_TARGETS = {1: 80, 2: 80, 3: 80, 4: 80, 5: 80, 6: 120, 7: 120, 8: 120, 9: 120, 10: 120, 11: 120, 12: 120}
VIEWS = {
    "front": {
        "type": "Perspective",
        "position": (0.0, -7.5, 3.2),
        "look_at": (0.0, 0.0, 0.35),
        "focal_length": 35,
        "sensor_width": 32,
    },
    "back": {
        "type": "Perspective",
        "position": (0.0, 7.5, 3.2),
        "look_at": (0.0, 0.0, 0.35),
        "focal_length": 35,
        "sensor_width": 32,
    },
    "right": {
        "type": "Perspective",
        "position": (7.5, 0.0, 3.2),
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
    parser.add_argument("--views", nargs="+", choices=sorted(VIEWS), default=["front", "back", "left", "right", "top"])
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
    size = (9.0, 6.0, 0.08)
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
    restitution: float = 0.8,
    lateral_friction: float = 0.0,
    color_name: str = "gray",
) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        object_type="wall",
        name=f"wall_{object_id}",
        static=True,
        size=size,
        position=position,
        mass=1.0,
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


def build_s7_stratified_level_configs(level_id: int, start_id: int, seed: int, count: int) -> list[SampleSpec]:
    rng = np.random.default_rng(seed + level_id * 1000)
    colors = ["red", "blue", "yellow", "green"]
    configs: list[dict[str, Any]] = []

    if level_id == 1:
        # Level 1: 恢复系数的影响 (e -> 能量传递效率与末速度)
        restitution_values = cycle_values([0.0, 0.3, 0.5, 0.8, 1.0], count, rng)
        x_positions = cycle_values([-0.95, -0.80, -0.65], count, rng)
        y_positions = cycle_values([-0.20, 0.0, 0.20], count, rng)
        color_values = cycle_values(colors, count, rng)
        for restitution, x, y, color in zip(restitution_values, x_positions, y_positions, color_values):
            obj_a = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (x, y, 0.22), (1.8, 0.0, 0.0), mass=1.0, restitution=1.0, lateral_friction=0.0, color_name=color)
            obj_b = sphere_spec(RELAY_OBJECT_ID, 0.22, (0.0, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=restitution, lateral_friction=0.0, color_name=color)
            obj_c = sphere_spec(TERMINAL_OBJECT_ID, 0.22, (0.46, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=restitution, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("restitution", "restitution_to_energy_transfer", "restitution_bc",
                                   (ground_spec(), obj_a, obj_b, obj_c)))

    elif level_id == 2:
        # Level 2: 入射速度大小的影响 (v0 -> 动能输入与输出关系)
        speed_values = cycle_values([1.4, 1.7, 2.0, 2.4, 2.8], count, rng)
        x_positions = cycle_values([-0.95, -0.80, -0.65], count, rng)
        y_positions = cycle_values([-0.20, 0.0, 0.20], count, rng)
        color_values = cycle_values(colors, count, rng)
        for speed, x, y, color in zip(speed_values, x_positions, y_positions, color_values):
            obj_a = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (x, y, 0.22), (speed, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_b = sphere_spec(RELAY_OBJECT_ID, 0.22, (0.0, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_c = sphere_spec(TERMINAL_OBJECT_ID, 0.22, (0.46, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("initial_velocity", "v0_to_energy_output", "speed",
                                   (ground_spec(), obj_a, obj_b, obj_c)))

    elif level_id == 3:
        # Level 3: 质量序列的影响 (质量比 -> 动量传递效率)
        mass_sequences = [
            (1.0, 1.0, 1.0),  # equal
            (0.7, 1.0, 1.6),  # increasing
            (1.6, 1.0, 0.7),  # decreasing
            (1.0, 2.5, 1.0),  # middle_heavy
            (1.0, 0.4, 1.0),  # middle_light
        ]
        speed_values = cycle_values([1.6, 2.2], count, rng)
        y_positions = cycle_values([-0.20, 0.0, 0.20], count, rng)
        color_values = cycle_values(colors, count, rng)
        for masses, speed, y, color in zip(
            cycle_values(mass_sequences, count, rng), speed_values, y_positions, color_values
        ):
            obj_a = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (-0.85, y, 0.22), (speed, 0.0, 0.0), mass=masses[0], restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_b = sphere_spec(RELAY_OBJECT_ID, 0.22, (0.0, y, 0.22), (0.0, 0.0, 0.0), mass=masses[1], restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_c = sphere_spec(TERMINAL_OBJECT_ID, 0.22, (0.46, y, 0.22), (0.0, 0.0, 0.0), mass=masses[2], restitution=0.8, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("mass_sequence", "mass_sequence_affects_momentum", "mass_sequence",
                                   (ground_spec(), obj_a, obj_b, obj_c)))

    elif level_id == 4:
        # Level 4: 尺寸分布的影响 (半径比 -> 接触几何与碰撞时序)
        radius_sequences = [
            (0.22, 0.22, 0.22),  # equal
            (0.18, 0.22, 0.28),  # increasing
            (0.28, 0.22, 0.18),  # decreasing
            (0.20, 0.28, 0.20),  # middle_large
            (0.26, 0.18, 0.26),  # middle_small
        ]
        gap_values = cycle_values([0.45, 0.60], count, rng)
        y_positions = cycle_values([-0.15, 0.0, 0.15], count, rng)
        color_values = cycle_values(colors, count, rng)
        for radii, gap, y, color in zip(
            cycle_values(radius_sequences, count, rng), gap_values, y_positions, color_values
        ):
            x_a = -(radii[0] + radii[1] + gap)
            x_c = radii[1] + radii[2] + 0.02
            obj_a = sphere_spec(PRIMARY_OBJECT_ID, radii[0], (x_a, y, radii[0]), (1.8, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_b = sphere_spec(RELAY_OBJECT_ID, radii[1], (0.0, y, radii[1]), (0.0, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_c = sphere_spec(TERMINAL_OBJECT_ID, radii[2], (x_c, y, radii[2]), (0.0, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("radius_sequence", "radius_affects_timing", "radius_sequence",
                                   (ground_spec(), obj_a, obj_b, obj_c)))

    elif level_id == 5:
        # Level 5: 初始间隙的影响 (间隙 -> 碰撞时序与二次碰撞)
        gap_ab_values = cycle_values([0.30, 0.45, 0.60], count, rng)
        gap_bc_values = cycle_values([0.0, 0.03, 0.08, 0.16], count, rng)
        speed_values = cycle_values([1.8, 2.4], count, rng)
        y_positions = cycle_values([-0.20, 0.0, 0.20], count, rng)
        color_values = cycle_values(colors, count, rng)
        for gap_ab, gap_bc, speed, y, color in zip(gap_ab_values, gap_bc_values, speed_values, y_positions, color_values):
            obj_a = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (-(0.44 + gap_ab), y, 0.22), (speed, 0.0, 0.0), mass=1.0, restitution=0.5, lateral_friction=0.0, color_name=color)
            obj_b = sphere_spec(RELAY_OBJECT_ID, 0.22, (0.0, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=0.5, lateral_friction=0.0, color_name=color)
            obj_c = sphere_spec(TERMINAL_OBJECT_ID, 0.22, (0.44 + gap_bc, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=0.5, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("gap", "gap_affects_timing", "gap_ab_gap_bc",
                                   (ground_spec(), obj_a, obj_b, obj_c)))

    elif level_id == 6:
        # Level 6: 物体类型序列的影响 (球/方块/圆柱排列)
        shape_sequences = [
            ("sphere", "sphere", "sphere"),
            ("sphere", "cube", "sphere"),
            ("sphere", "cube", "cube"),
            ("cube", "sphere", "cube"),
            ("sphere", "cylinder", "sphere"),
            ("cylinder", "sphere", "cylinder"),
        ]
        y_positions = cycle_values([-0.15, 0.0, 0.15], count, rng)
        color_values = cycle_values(colors, count, rng)

        def contact_radius(shape: str) -> float:
            if shape == "sphere":
                return 0.22
            if shape == "cube":
                return 0.12
            return 0.20

        def contact_height(shape: str) -> float:
            if shape == "sphere":
                return 0.22
            if shape == "cube":
                return 0.12
            return 0.14

        for shapes, y, color in zip(cycle_values(shape_sequences, count, rng), y_positions, color_values):
            objects = []
            radii = [contact_radius(shape) for shape in shapes]
            x_offsets = [-(radii[0] + radii[1] + 0.50), 0.0, radii[1] + radii[2] + 0.04]
            for i, (shape, x_off) in enumerate(zip(shapes, x_offsets)):
                obj_id = PRIMARY_OBJECT_ID + i
                z = contact_height(shape)
                velocity = (2.0, 0.0, 0.0) if i == 0 else (0.0, 0.0, 0.0)
                if shape == "sphere":
                    obj = sphere_spec(obj_id, 0.22, (x_off, y, z), velocity, mass=1.0, restitution=0.8, lateral_friction=0.5, color_name=color)
                elif shape == "cube":
                    obj = cube_spec(obj_id, (0.24, 0.24, 0.24), (x_off, y, z), velocity, mass=1.0, restitution=0.8, lateral_friction=0.5, color_name=color)
                else:  # cylinder
                    obj = cylinder_spec(obj_id, 0.20, 0.28, (x_off, y, z), velocity, mass=1.0, restitution=0.8, lateral_friction=0.5, color_name=color)
                objects.append(obj)
            configs.append(make_cfg("shape_sequence", "shape_affects_chain", "shape_sequence",
                                   (ground_spec(), *objects)))

    elif level_id == 7:
        # Level 7: 二维空间排列与非对心碰撞 (横向偏移 -> 散射角度与能量分配)
        offset_sequences = [
            (0.0, 0.0),
            (0.10, 0.0),
            (-0.10, 0.0),
            (0.0, 0.10),
            (0.12, -0.12),
            (-0.12, 0.12),
        ]
        speed_values = cycle_values([2.0, 2.6], count, rng)
        x_positions = cycle_values([-0.90, -0.75], count, rng)
        color_values = cycle_values(colors, count, rng)
        for offsets, speed, x, color in zip(cycle_values(offset_sequences, count, rng), speed_values, x_positions, color_values):
            obj_a = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (x, 0.0, 0.22), (speed, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_b = sphere_spec(RELAY_OBJECT_ID, 0.22, (0.0, offsets[0], 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_c = sphere_spec(TERMINAL_OBJECT_ID, 0.22, (0.46, offsets[1], 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("spatial_offset", "offset_affects_scattering", "offset_sequence",
                                   (ground_spec(), obj_a, obj_b, obj_c)))

    elif level_id == 8:
        # Level 8: 多物体具有初始速度 (追尾/对撞 -> 相对速度效应)
        velocity_modes = {
            "baseline": ((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "chase_mid": ((2.4, 0.0, 0.0), (0.6, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "chase_all": ((2.6, 0.0, 0.0), (0.8, 0.0, 0.0), (0.2, 0.0, 0.0)),
            "mid_opposite": ((1.8, 0.0, 0.0), (-0.4, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "terminal_opposite": ((1.8, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.5, 0.0, 0.0)),
            "terminal_escape": ((2.4, 0.0, 0.0), (0.0, 0.0, 0.0), (0.7, 0.0, 0.0)),
        }
        mode_values = cycle_values(list(velocity_modes), count, rng)
        x_positions = cycle_values([-0.95, -0.80], count, rng)
        y_positions = cycle_values([-0.15, 0.0, 0.15], count, rng)
        color_values = cycle_values(colors, count, rng)
        for mode, x, y, color in zip(mode_values, x_positions, y_positions, color_values):
            vel_a, vel_b, vel_c = velocity_modes[mode]
            obj_a = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (x, y, 0.22), vel_a, mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_b = sphere_spec(RELAY_OBJECT_ID, 0.22, (0.0, y, 0.22), vel_b, mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_c = sphere_spec(TERMINAL_OBJECT_ID, 0.22, (0.52, y, 0.22), vel_c, mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("multi_velocity", "relative_velocity_effect", "velocity_mode",
                                   (ground_spec(), obj_a, obj_b, obj_c)))

    elif level_id == 9:
        # Level 9: 初始旋转的影响 (角速度 -> 切向摩擦与旋转传递)
        spin_modes = cycle_values(["no_spin", "spin_z_pos", "spin_z_neg", "spin_y_pos", "spin_y_neg"], count, rng)
        x_positions = cycle_values([-0.90, -0.75], count, rng)
        y_offsets = cycle_values([-0.10, 0.10], count, rng)
        color_values = cycle_values(colors, count, rng)
        for spin, x, y, color in zip(spin_modes, x_positions, y_offsets, color_values):
            if spin == "no_spin":
                angular = (0.0, 0.0, 0.0)
            elif spin == "spin_z_pos":
                angular = (0.0, 0.0, 8.0)
            elif spin == "spin_z_neg":
                angular = (0.0, 0.0, -8.0)
            elif spin == "spin_y_pos":
                angular = (0.0, 8.0, 0.0)
            else:
                angular = (0.0, -8.0, 0.0)
            obj_a = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (x, 0.0, 0.22), (2.2, 0.0, 0.0), angular_velocity=angular, mass=1.0, restitution=0.8, lateral_friction=1.0, color_name=color)
            obj_b = sphere_spec(RELAY_OBJECT_ID, 0.22, (0.0, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=1.0, color_name=color)
            obj_c = sphere_spec(TERMINAL_OBJECT_ID, 0.22, (0.46, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("initial_spin", "spin_affects_chain_transfer", "object_a.initial_angular_velocity",
                                   (ground_spec(), obj_a, obj_b, obj_c)))

    elif level_id == 10:
        # Level 10: 边界约束下的连锁碰撞 (靠墙反射 -> 多次碰撞与能量耗散)
        wall_modes = cycle_values(["right_wall", "both_walls"], count, rng)
        restitution_values = cycle_values([0.5, 0.8, 1.0], count, rng)
        speed_values = cycle_values([2.0, 2.6], count, rng)
        y_positions = cycle_values([-0.15, 0.0, 0.15], count, rng)
        color_values = cycle_values(colors, count, rng)
        wall_colors = cycle_values(["gray", "blue"], count, rng)
        wall_size = (0.08, 5.00, 1.20)
        for mode, restitution, speed, y, color, wall_color in zip(wall_modes, restitution_values, speed_values, y_positions, color_values, wall_colors):
            walls = [wall_spec(8, wall_size, (1.35, 0.0, 0.60), restitution=restitution, color_name=wall_color)]
            if mode == "both_walls":
                walls.append(wall_spec(9, wall_size, (-1.35, 0.0, 0.60), restitution=restitution, color_name="blue" if wall_color == "gray" else "gray"))
            obj_a = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (-0.95, y, 0.22), (speed, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_b = sphere_spec(RELAY_OBJECT_ID, 0.22, (-0.05, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_c = sphere_spec(TERMINAL_OBJECT_ID, 0.22, (0.41, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("wall_constraint", "wall_reflection_in_chain", "wall_mode_restitution",
                                   (ground_spec(), *walls, obj_a, obj_b, obj_c)))

    elif level_id == 11:
        # Level 11: 物体材质异质性的影响 (各物体恢复系数不同 -> 非对称能量分配)
        sequences = [
            (1.0, 1.0, 1.0),
            (1.0, 0.3, 1.0),
            (1.0, 1.0, 0.3),
            (0.3, 1.0, 1.0),
            (0.3, 0.8, 1.0),
            (1.0, 0.8, 0.3),
        ]
        speed_values = cycle_values([1.8, 2.4], count, rng)
        x_positions = cycle_values([-0.90, -0.75], count, rng)
        y_positions = cycle_values([-0.20, 0.0, 0.20], count, rng)
        color_values = cycle_values(colors, count, rng)
        for restitution_seq, speed, x, y, color in zip(cycle_values(sequences, count, rng), speed_values, x_positions, y_positions, color_values):
            obj_a = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (x, y, 0.22), (speed, 0.0, 0.0), mass=1.0, restitution=restitution_seq[0], lateral_friction=0.0, color_name=color)
            obj_b = sphere_spec(RELAY_OBJECT_ID, 0.22, (0.0, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=restitution_seq[1], lateral_friction=0.0, color_name=color)
            obj_c = sphere_spec(TERMINAL_OBJECT_ID, 0.22, (0.46, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=restitution_seq[2], lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("restitution_sequence", "heterogeneous_material_chain", "restitution_sequence",
                                   (ground_spec(), obj_a, obj_b, obj_c)))

    elif level_id == 12:
        # Level 12: 地面摩擦存在下的连锁碰撞 (碰撞后滑行与停止)
        friction_values = cycle_values([0.05, 0.15, 0.30, 0.50, 0.80], count, rng)
        speed_values = cycle_values([1.8, 2.4], count, rng)
        x_positions = cycle_values([-0.95, -0.80], count, rng)
        y_positions = cycle_values([-0.20, 0.0, 0.20], count, rng)
        color_values = cycle_values(colors, count, rng)
        for friction, speed, x, y, color in zip(friction_values, speed_values, x_positions, y_positions, color_values):
            obj_a = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (x, y, 0.22), (speed, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=1.0, color_name=color)
            obj_b = sphere_spec(RELAY_OBJECT_ID, 0.22, (0.0, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=1.0, color_name=color)
            obj_c = sphere_spec(TERMINAL_OBJECT_ID, 0.22, (0.46, y, 0.22), (0.0, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=1.0, color_name=color)
            configs.append(make_cfg("ground_friction", "friction_affects_chain", "ground.lateralFriction",
                                   (ground_spec(restitution=0.0, friction=friction), obj_a, obj_b, obj_c)))

    else:
        raise ValueError(f"S7 supports levels 1..12, got L{level_id}")

    return with_ids(configs, level_id, start_id, seed, shuffle=False)


def target_count_for_level(level_id: int, requested_count: int | None) -> int:
    if requested_count is not None:
        return requested_count
    if level_id not in LEVEL_TARGETS:
        raise ValueError(f"S7 supports levels 1..12, got L{level_id}")
    return LEVEL_TARGETS[level_id]


def take_samples(level_id: int, start_id: int, seed: int, count: int | None) -> list[SampleSpec]:
    target_count = target_count_for_level(level_id, count)
    return build_s7_stratified_level_configs(level_id, start_id, seed, target_count)


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
    raise ValueError(f"Unsupported object type for S7: {spec.object_type}")


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
