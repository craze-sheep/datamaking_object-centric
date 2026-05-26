#!/usr/bin/env python3
"""Generate S8 horizontal sliding samples with Kubric.

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


class FilteredNegativeSample(Exception):
    """Raised when a candidate violates S8's no dynamic-dynamic collision rule."""


FPS = 12
NUM_FRAMES = 36
SIM_HZ = 240
DURATION_S = 3.0
RESOLUTION = 128
GRAVITY = (0.0, 0.0, -9.8)
SCENE_ID = 8
GROUND_OBJECT_ID = 1
PRIMARY_OBJECT_ID = 2
LEVEL_TARGETS = {1: 120, 2: 120, 3: 120, 4: 120, 5: 120, 6: 120, 7: 120, 8: 120, 9: 120, 10: 120, 11: 120, 12: 120, 13: 120, 14: 120}
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

DYNAMIC_CUBE_SIZES = [
    (0.18, 0.18, 0.18),
    (0.20, 0.20, 0.20),
    (0.24, 0.24, 0.24),
    (0.26, 0.26, 0.26),
    (0.30, 0.30, 0.30),
]
OBSTACLE_SIZES = [
    (0.20, 1.20, 0.80),
    (1.20, 0.20, 0.80),
    (0.30, 0.80, 0.80),
]
VERTICAL_WALL_SIZE = (0.08, 5.00, 1.20)
HORIZONTAL_WALL_SIZE = (5.00, 0.08, 1.20)


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
    metadata: dict[str, Any] | None = None


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
    restitution: float = 0.9,
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


def obstacle_spec(
    object_id: int,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    restitution: float = 0.5,
    lateral_friction: float = 0.5,
    color_name: str = "gray",
) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        object_type="obstacle",
        name=f"obstacle_{object_id}",
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
                metadata=cfg.get("metadata"),
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


def contact_radius_2d(spec: ObjectSpec) -> float:
    if spec.radius is not None:
        return float(spec.radius)
    if spec.size is not None:
        return 0.5 * float(math.hypot(spec.size[0], spec.size[1]))
    return 0.0


def initial_clearance(a: ObjectSpec, b: ObjectSpec) -> float:
    pa = np.array(a.position[:2], dtype=np.float64)
    pb = np.array(b.position[:2], dtype=np.float64)
    return float(np.linalg.norm(pa - pb) - contact_radius_2d(a) - contact_radius_2d(b))


def has_initial_overlap(objects: list[ObjectSpec], margin: float = 0.0) -> bool:
    dynamic_objects = [obj for obj in objects if not obj.static]
    for idx, obj_a in enumerate(dynamic_objects):
        for obj_b in dynamic_objects[idx + 1:]:
            if initial_clearance(obj_a, obj_b) < margin:
                return True
    return False


def make_cfg(level_name: str, subtask: str, main_variable: str, objects: tuple[ObjectSpec, ...], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "level_name": level_name,
        "subtask": subtask,
        "main_variable": main_variable,
        "objects": objects,
        "metadata": metadata,
    }


def build_s8_stratified_level_configs(level_id: int, start_id: int, seed: int, count: int) -> list[SampleSpec]:
    rng = np.random.default_rng(seed + level_id * 1000)
    colors = ["red", "blue", "yellow", "green"]
    configs: list[dict[str, Any]] = []

    if level_id == 1:
        # Level 1: 空间分离导致的未碰撞 (平行轨迹错开)
        object_counts = cycle_values([2, 3, 4, 5, 6], count, rng)
        lane_y_values = [-1.50, -0.90, -0.30, 0.30, 0.90, 1.50]
        speed_values = cycle_values([0.8, 1.2, 1.6, 2.0], count, rng)
        direction_values = cycle_values([(1, 0, 0), (-1, 0, 0)], count, rng)
        color_values = cycle_values(colors, count, rng)
        for obj_count, speed, direction, color in zip(object_counts, speed_values, direction_values, color_values):
            objects = []
            lanes = rng.choice(lane_y_values, size=obj_count, replace=False)
            for i in range(obj_count):
                shape = pick(["sphere", "cube"], rng)
                x = float(pick([-1.8, -1.4, -1.0], rng))
                velocity = (speed * direction[0], speed * direction[1], 0.0)
                if shape == "sphere":
                    radius = float(pick([0.18, 0.22, 0.28], rng))
                    obj = sphere_spec(PRIMARY_OBJECT_ID + i, radius, (x, lanes[i], radius), velocity,
                                     mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
                else:
                    size = pick(DYNAMIC_CUBE_SIZES, rng)
                    obj = cube_spec(PRIMARY_OBJECT_ID + i, size, (x, lanes[i], size[2] / 2), velocity,
                                   mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
                objects.append(obj)
            configs.append(make_cfg("spatial_separation", "parallel_trajectories_no_collision", "lane_y",
                                   (ground_spec(), *objects)))

    elif level_id == 2:
        # Level 2: 时间分离导致的未碰撞 (先后经过同一点)
        object_counts = cycle_values([2, 3, 4, 5], count, rng)
        speed_values = cycle_values([0.8, 1.1, 1.5, 2.0], count, rng)
        color_values = cycle_values(colors, count, rng)
        path_directions = [
            (1.0, 0.0, 0.0),
            (0.309017, 0.951057, 0.0),
            (-0.809017, 0.587785, 0.0),
            (-0.809017, -0.587785, 0.0),
            (0.309017, -0.951057, 0.0),
        ]
        for obj_count, speed, color in zip(object_counts, speed_values, color_values):
            objects = []
            for i in range(obj_count):
                direction = path_directions[i % len(path_directions)]
                distance = 1.8 + i * 1.5 * speed
                x = -distance * direction[0]
                y = -distance * direction[1]
                obj = sphere_spec(PRIMARY_OBJECT_ID + i, 0.22, (x, y, 0.22),
                                 (speed * direction[0], speed * direction[1], 0.0),
                                 mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
                objects.append(obj)
            configs.append(make_cfg("temporal_separation", "staggered_arrival_no_collision", "start_delay",
                                   (ground_spec(), *objects)))

    elif level_id == 3:
        # Level 3: 速度差异导致的未碰撞 (追不上或已超越)
        modes = cycle_values(["cannot_catch_up", "crossing_time_offset"], count, rng)
        catchup_counts = cycle_values([2, 3, 4], count, rng)
        crossing_counts = cycle_values([2, 3], count, rng)
        front_speeds = cycle_values([1.4, 1.8, 2.2], count, rng)
        rear_speeds = cycle_values([0.4, 0.8, 1.2], count, rng)
        y_positions = cycle_values([-0.25, 0.0, 0.25], count, rng)
        crossing_speeds = cycle_values([2.0, 2.4], count, rng)
        color_values = cycle_values(colors, count, rng)
        x_positions = [-1.6, -0.6, 0.4, 1.2]
        path_directions = [(1.0, 0.0), (-0.5, 0.8660254), (-0.5, -0.8660254)]
        for idx, (mode, front_speed, rear_speed, y, crossing_speed, color) in enumerate(zip(modes, front_speeds, rear_speeds, y_positions, crossing_speeds, color_values)):
            objects = []
            if mode == "cannot_catch_up":
                obj_count = catchup_counts[idx]
                for i in range(obj_count):
                    shape = pick(["sphere", "cube"], rng)
                    x = x_positions[i]
                    speed = rear_speed if i == 0 else front_speed
                    if shape == "sphere":
                        obj = sphere_spec(PRIMARY_OBJECT_ID + i, 0.22, (x, y, 0.22),
                                         (speed, 0.0, 0.0),
                                         mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
                    else:
                        obj = cube_spec(PRIMARY_OBJECT_ID + i, (0.24, 0.24, 0.24), (x, y, 0.12),
                                       (speed, 0.0, 0.0),
                                       mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
                    objects.append(obj)
                configs.append(make_cfg("speed_difference", "cannot_catch_up", "speed_ratio",
                                       (ground_spec(), *objects),
                                       metadata={"subscene": "A", "initial_gap_min": 0.80}))
            else:
                obj_count = crossing_counts[idx]
                for i in range(obj_count):
                    direction = np.array(path_directions[i], dtype=np.float64)
                    speed = crossing_speed if i == 0 else float(pick([2.0, 2.4], rng))
                    arrival_time = 0.75 + i * 0.55
                    start_xy = -direction * speed * arrival_time
                    objects.append(sphere_spec(PRIMARY_OBJECT_ID + i, 0.22, (float(start_xy[0]), float(start_xy[1]), 0.22),
                                               (float(speed * direction[0]), float(speed * direction[1]), 0.0),
                                               mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color))
                configs.append(make_cfg("speed_difference", "crossing_paths_time_offset", "arrival_time_gap",
                                       (ground_spec(), *objects),
                                       metadata={"subscene": "B", "arrival_time_gap_min": 0.55}))

    elif level_id == 4:
        # Level 4: 随机参数全空间的未碰撞过滤
        object_counts = cycle_values([1, 2, 3, 4, 5, 6], count, rng)
        ground_frictions = cycle_values([0.0, 0.1, 0.3, 0.5], count, rng)
        color_values = cycle_values(colors, count, rng)
        for obj_count, ground_friction, color in zip(object_counts, ground_frictions, color_values):
            objects = []
            for i in range(obj_count):
                shape = pick(["sphere", "cube", "cylinder"], rng)
                x = rng.uniform(-2.5, 2.5)
                y = rng.uniform(-1.8, 1.8)
                vx = rng.uniform(-2.0, 2.0)
                vy = rng.uniform(-1.5, 1.5)
                mass = rng.choice([0.5, 1.0, 2.0])
                restitution = float(rng.choice([0.3, 0.5, 0.8, 1.0]))
                friction = float(rng.choice([0.0, 0.3, 0.5, 0.8, 1.0]))
                angular = (float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4)))
                if shape == "sphere":
                    radius = float(rng.choice([0.18, 0.22, 0.28]))
                    obj = sphere_spec(PRIMARY_OBJECT_ID + i, radius, (x, y, radius), (vx, vy, 0.0), angular_velocity=angular,
                                     mass=mass, restitution=restitution, lateral_friction=friction, color_name=color)
                elif shape == "cube":
                    size = pick(DYNAMIC_CUBE_SIZES, rng)
                    obj = cube_spec(PRIMARY_OBJECT_ID + i, size, (x, y, size[2] / 2), (vx, vy, 0.0), angular_velocity=angular,
                                   mass=mass, restitution=restitution, lateral_friction=friction, color_name=color)
                else:
                    radius = float(rng.choice([0.16, 0.20, 0.24]))
                    height = float(rng.choice([0.22, 0.28, 0.34]))
                    obj = cylinder_spec(PRIMARY_OBJECT_ID + i, radius, height, (x, y, height / 2), (vx, vy, 0.0), angular_velocity=angular,
                                       mass=mass, restitution=restitution, lateral_friction=friction, color_name=color)
                objects.append(obj)
            configs.append(make_cfg("random_space", "random_no_collision", "object_count",
                                   (ground_spec(friction=ground_friction), *objects)))

    elif level_id == 5:
        # Level 5: 物体被静态障碍物阻隔导致未碰撞
        object_counts = cycle_values([2, 3, 4, 5], count, rng)
        obstacle_sizes = cycle_values([(0.30, 3.20, 0.80), (0.40, 3.00, 0.80)], count, rng)
        obstacle_restitutions = cycle_values([0.5, 0.8], count, rng)
        obstacle_y_values = cycle_values([-0.30, 0.0, 0.30], count, rng)
        speed_values = cycle_values([0.8, 1.2, 1.6], count, rng)
        color_values = cycle_values(colors, count, rng)
        for obj_count, obs_size, obs_restitution, obs_y, speed, color in zip(object_counts, obstacle_sizes, obstacle_restitutions, obstacle_y_values, speed_values, color_values):
            objects = []
            obs = obstacle_spec(9, obs_size, (0.0, obs_y, obs_size[2] / 2), restitution=obs_restitution, color_name="gray")
            objects.append(obs)
            for i in range(obj_count):
                shape = pick(["sphere", "cube"], rng)
                side = 1 if i % 2 == 0 else -1
                x = side * rng.uniform(1.0, 2.0)
                y = obs_y + [-1.20, -0.60, 0.0, 0.60, 1.20][i]
                vx = -side * speed
                if shape == "sphere":
                    radius = float(pick([0.18, 0.22, 0.28], rng))
                    obj = sphere_spec(PRIMARY_OBJECT_ID + i, radius, (x, y, radius),
                                     (vx, 0.0, 0.0),
                                     mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
                else:
                    size = pick(DYNAMIC_CUBE_SIZES, rng)
                    obj = cube_spec(PRIMARY_OBJECT_ID + i, size, (x, y, size[2] / 2),
                                   (vx, 0.0, 0.0),
                                   mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
                objects.append(obj)
            configs.append(make_cfg("blocked_by_obstacle", "obstacle_blocks_path", "obstacle_size",
                                   (ground_spec(), *objects)))

    elif level_id == 6:
        # Level 6: 物体在碰撞前因摩擦力停止运动
        object_counts = cycle_values([2, 3, 4], count, rng)
        friction_values = cycle_values([0.08, 0.15, 0.25, 0.40, 0.60], count, rng)
        speed_values = cycle_values([0.8, 1.0, 1.2, 1.5], count, rng)
        color_values = cycle_values(colors, count, rng)
        for obj_count, friction, speed, color in zip(object_counts, friction_values, speed_values, color_values):
            objects = []
            positions = [(-2.2, -0.4), (-1.6, 0.0), (1.6, 0.4), (2.2, -0.4)]
            for i in range(obj_count):
                shape = pick(["cube", "sphere"], rng)
                x, y = positions[i]
                direction = 1 if x < 0 else -1
                mass = float(pick([1.0, 2.0], rng))
                velocity = (direction * speed, 0.0, 0.0)
                if shape == "sphere":
                    radius = float(pick([0.18, 0.22, 0.28], rng))
                    obj = sphere_spec(PRIMARY_OBJECT_ID + i, radius, (x, y, radius), velocity,
                                     mass=mass, restitution=0.5, lateral_friction=1.0, color_name=color)
                else:
                    size = pick(DYNAMIC_CUBE_SIZES, rng)
                    obj = cube_spec(PRIMARY_OBJECT_ID + i, size, (x, y, size[2] / 2), velocity,
                                   mass=mass, restitution=0.5, lateral_friction=1.0, color_name=color)
                objects.append(obj)
            configs.append(make_cfg("friction_stop", "stop_before_collision", "lateralFriction",
                                   (ground_spec(restitution=0.0, friction=friction), *objects)))

    elif level_id == 7:
        # Level 7: 物体轨迹在三维空间高度上错开
        object_counts = cycle_values([2, 3, 4, 5], count, rng)
        speed_values = cycle_values([0.30, 0.40, 0.50], count, rng)
        vz_values = cycle_values([-0.4, 0.0, 0.4], count, rng)
        color_values = cycle_values(colors, count, rng)
        height_pool = [0.22, 0.95, 1.70, 2.45, 3.20]
        lanes = [(-1.80, -1.20), (-0.90, -0.60), (0.0, 0.0), (0.90, 0.60), (1.80, 1.20)]
        for obj_count, speed, vz, color in zip(object_counts, speed_values, vz_values, color_values):
            objects = []
            z_values = rng.choice(height_pool, size=obj_count, replace=False)
            for i in range(obj_count):
                radius = float(pick([0.18, 0.22], rng))
                x, y = lanes[i]
                z = float(z_values[i])
                obj = sphere_spec(PRIMARY_OBJECT_ID + i, radius, (x, y, z),
                                 (speed, 0.0, vz),
                                 mass=1.0, restitution=0.5, lateral_friction=0.0, color_name=color)
                objects.append(obj)
            configs.append(make_cfg("height_separation", "height_and_lane_separation_no_collision", "z_and_lane_separation",
                                   (ground_spec(), *objects),
                                   metadata={"separation_factors": ["z_position", "xy_lane"]}))

    elif level_id == 8:
        # Level 8: 物体被其他运动物体阻挡但未直接碰撞
        color_values = cycle_values(colors, count, rng)
        speed_values = cycle_values([0.8, 1.0], count, rng)
        for speed, color in zip(speed_values, color_values):
            obj_left = sphere_spec(PRIMARY_OBJECT_ID, 0.22, (-1.40, 0.0, 0.22), (speed, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_shield = sphere_spec(RELAY_OBJECT_ID if 'RELAY_OBJECT_ID' in dir() else 3, 0.22, (0.0, 0.0, 0.22), (speed, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            obj_right = sphere_spec(TERMINAL_OBJECT_ID if 'TERMINAL_OBJECT_ID' in dir() else 4, 0.22, (1.40, 0.0, 0.22), (speed, 0.0, 0.0), mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color)
            configs.append(make_cfg("shielded_by_object", "moving_shield_no_collision", "speed",
                                   (ground_spec(), obj_left, obj_shield, obj_right)))

    elif level_id == 9:
        # Level 9: 物体在反弹边界内反复运动但总错过
        object_counts = cycle_values([2, 3, 4], count, rng)
        speed_values = cycle_values([1.0, 1.4, 1.8], count, rng)
        angle_values = cycle_values([20, 35, 55, 125, 145, 200, 235, 305], count, rng)
        wall_restitutions = cycle_values([0.9, 1.0], count, rng)
        wall_sets = cycle_values(["box_boundary", "half_box_boundary"], count, rng)
        color_values = cycle_values(colors, count, rng)
        wall_size_v = VERTICAL_WALL_SIZE
        wall_size_h = HORIZONTAL_WALL_SIZE
        for obj_count, speed, angle_deg, wall_restitution, wall_set, color in zip(object_counts, speed_values, angle_values, wall_restitutions, wall_sets, color_values):
            objects = []
            # Add boundary walls
            objects.append(wall_spec(6, wall_size_v, (-2.2, 0.0, 0.60), restitution=wall_restitution, color_name="gray"))
            objects.append(wall_spec(7, wall_size_v, (2.2, 0.0, 0.60), restitution=wall_restitution, color_name="blue"))
            if wall_set == "box_boundary":
                objects.append(wall_spec(8, wall_size_h, (0.0, 1.6, 0.60), restitution=wall_restitution, color_name="gray"))
            objects.append(wall_spec(9, wall_size_h, (0.0, -1.6, 0.60), restitution=wall_restitution, color_name="blue"))
            for i in range(obj_count):
                radius = float(pick([0.18, 0.22], rng))
                angle_rad = math.radians(angle_deg + i * 90)
                x = 1.0 * math.cos(angle_rad)
                y = 0.8 * math.sin(angle_rad)
                vx = speed * math.cos(angle_rad + math.pi / 2)
                vy = speed * math.sin(angle_rad + math.pi / 2)
                obj = sphere_spec(PRIMARY_OBJECT_ID + i, radius, (x, y, radius),
                                 (vx, vy, 0.0),
                                 mass=1.0, restitution=1.0, lateral_friction=0.0, color_name=color)
                objects.append(obj)
            configs.append(make_cfg("boundary_bounce", "bounce_and_miss", "direction_angle",
                                   (ground_spec(), *objects),
                                   metadata={"wall_set": wall_set}))

    elif level_id == 10:
        # Level 10: 物体尺寸极小或碰撞边界余量设置导致"擦肩"
        object_counts = cycle_values([2, 3, 4], count, rng)
        radius_values = cycle_values([0.12, 0.16, 0.18], count, rng)
        stress_values = cycle_values([False] * 19 + [True], count, rng)
        margin_values = cycle_values([0.0, 0.005, 0.01], count, rng)
        speed_values = cycle_values([0.8, 1.2, 1.6], count, rng)
        y_offsets = cycle_values([-0.42, 0.42, 0.50, -0.50], count, rng)
        color_values = cycle_values(colors, count, rng)
        for obj_count, radius, stress, margin, speed, y_off, color in zip(object_counts, radius_values, stress_values, margin_values, speed_values, y_offsets, color_values):
            if stress:
                radius = float(pick([0.05, 0.08], rng))
            objects = []
            for i in range(obj_count):
                side = 1 if i % 2 == 0 else -1
                x = side * 1.2
                y = y_off * (i + 1)
                vx = -side * speed
                mass = float(pick([0.3, 0.5, 1.0], rng))
                obj = sphere_spec(PRIMARY_OBJECT_ID + i, radius, (x, y, radius),
                                 (vx, 0.0, 0.0),
                                 mass=mass, restitution=0.8, lateral_friction=0.0, color_name=color)
                objects.append(obj)
            configs.append(make_cfg("near_miss", "close_pass_no_collision", "safety_margin",
                                   (ground_spec(), *objects),
                                   metadata={"collisionMargin": margin, "sample_split": "stress" if stress else "train", "safety_margin": 0.02 + margin}))

    elif level_id == 11:
        # Level 11: 旋转导致的接触面错开 (方块/圆柱)
        object_counts = cycle_values([2, 3], count, rng)
        shape_values = cycle_values(["cube", "cylinder"], count, rng)
        angular_modes = cycle_values([(8.0, 0.0, 0.0), (0.0, 0.0, 8.0), (4.0, 0.0, 6.0)], count, rng)
        speed_values = cycle_values([0.8, 1.2, 1.6], count, rng)
        ground_frictions = cycle_values([0.2, 0.5], count, rng)
        object_frictions = cycle_values([0.5, 1.0], count, rng)
        restitution_values = cycle_values([0.6, 0.8], count, rng)
        color_values = cycle_values(colors, count, rng)
        for obj_count, shape, angular, speed, ground_friction, object_friction, restitution, color in zip(object_counts, shape_values, angular_modes, speed_values, ground_frictions, object_frictions, restitution_values, color_values):
            objects = []
            for i in range(obj_count):
                x = -1.6 + i * 1.6
                y = [-0.75, 0.75, 1.75][i % 3]
                vx = speed if i == 0 else -speed * 0.5
                if shape == "cube":
                    size = pick([(0.18, 0.18, 0.18), (0.24, 0.24, 0.24)], rng)
                    quat = (1.0, 0.0, 0.0, 0.0) if i % 2 == 0 else (0.9239, 0.0, 0.0, 0.3827)
                    obj = cube_spec(PRIMARY_OBJECT_ID + i, size, (x, y, size[2] / 2), (vx, 0.0, 0.0),
                                   angular_velocity=angular, quaternion=quat,
                                   mass=1.0, restitution=restitution, lateral_friction=object_friction, color_name=color)
                else:
                    radius = pick([0.16, 0.20], rng)
                    height = pick([0.22, 0.28], rng)
                    obj = cylinder_spec(PRIMARY_OBJECT_ID + i, radius, height, (x, y, radius), (vx, 0.0, 0.0),
                                       angular_velocity=angular, quaternion=(0.7071, 0.0, 0.7071, 0.0),
                                       mass=1.0, restitution=restitution, lateral_friction=object_friction, color_name=color)
                objects.append(obj)
            configs.append(make_cfg("rotation_miss", "rotation_causes_miss", "angular_velocity",
                                   (ground_spec(friction=ground_friction), *objects)))

    elif level_id == 12:
        # Level 12: 力或冲量干预使物体在碰撞前转向
        object_counts = cycle_values([2], count, rng)
        speed_values = cycle_values([0.8, 1.0, 1.2], count, rng)
        apply_times = cycle_values([0.20, 0.30, 0.40], count, rng)
        impulses = cycle_values([(0.0, 1.20, 0.0), (0.0, -1.20, 0.0)], count, rng)
        duration_values = cycle_values([1, 2], count, rng)
        target_values = cycle_values([PRIMARY_OBJECT_ID, PRIMARY_OBJECT_ID + 1], count, rng)
        color_values = cycle_values(colors, count, rng)
        for obj_count, speed, apply_time, impulse, duration, target_object, color in zip(object_counts, speed_values, apply_times, impulses, duration_values, target_values, color_values):
            objects = []
            starts = [(-1.8, 0.0), (1.8, 0.0), (0.0, -1.8)]
            for i in range(obj_count):
                x, y = starts[i]
                direction = np.array([-x, -y], dtype=np.float64)
                direction = direction / np.linalg.norm(direction)
                vel = (float(speed * direction[0]), float(speed * direction[1]), 0.0)
                objects.append(sphere_spec(PRIMARY_OBJECT_ID + i, 0.22, (x, y, 0.22), vel,
                                           mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color))
            configs.append(make_cfg("external_impulse", "impulse_avoids_collision", "external_impulse",
                                   (ground_spec(), *objects),
                                   metadata={"external_impulse_enabled": True, "intervention_time": apply_time, "target_object": target_object, "impulse_vector": impulse, "duration_frames": duration, "counterfactual_would_collide": True}))

    elif level_id == 13:
        # Level 13: 完全静止的多物体场景
        object_counts = cycle_values([2, 3, 4, 5, 6], count, rng)
        color_values = cycle_values(colors, count, rng)
        for obj_count, color in zip(object_counts, color_values):
            objects = []
            for i in range(obj_count):
                for _ in range(200):
                    x = float(rng.uniform(-2.0, 2.0))
                    y = float(rng.uniform(-1.4, 1.4))
                    shape = ["sphere", "cube", "cylinder"][i % 3]
                    mass = float(pick([0.5, 1.0, 2.0], rng))
                    if shape == "sphere":
                        candidate = sphere_spec(PRIMARY_OBJECT_ID + i, 0.22, (x, y, 0.22), (0.0, 0.0, 0.0), mass=mass, restitution=0.5, lateral_friction=0.5, color_name=color)
                    elif shape == "cube":
                        size = pick(DYNAMIC_CUBE_SIZES, rng)
                        candidate = cube_spec(PRIMARY_OBJECT_ID + i, size, (x, y, size[2] / 2), (0.0, 0.0, 0.0), mass=mass, restitution=0.5, lateral_friction=0.5, color_name=color)
                    else:
                        candidate = cylinder_spec(PRIMARY_OBJECT_ID + i, 0.20, 0.28, (x, y, 0.14), (0.0, 0.0, 0.0), mass=mass, restitution=0.5, lateral_friction=0.5, color_name=color)
                    if all(initial_clearance(candidate, existing) >= 0.18 for existing in objects):
                        objects.append(candidate)
                        break
                else:
                    raise RuntimeError("Could not place non-overlapping static S8/L13 objects")
            configs.append(make_cfg("static_multi_object", "all_objects_static", "object_count",
                                   (ground_spec(friction=0.5), *objects),
                                   metadata={"total_contact_count_expected": 0}))

    elif level_id == 14:
        # Level 14: 速度方向严格背离的多个物体
        object_counts = cycle_values([3, 4, 5, 6], count, rng)
        cluster_radii = cycle_values([0.20, 0.35, 0.50], count, rng)
        speed_values = cycle_values([0.8, 1.2, 1.6, 2.0], count, rng)
        color_values = cycle_values(colors, count, rng)
        for obj_count, cluster_radius, speed, color in zip(object_counts, cluster_radii, speed_values, color_values):
            objects = []
            actual_cluster_radius = float(cluster_radius)
            for _ in range(20):
                trial_objects = []
                for i in range(obj_count):
                    angle = 2 * math.pi * i / obj_count
                    x = actual_cluster_radius * math.cos(angle)
                    y = actual_cluster_radius * math.sin(angle)
                    vx = speed * math.cos(angle)
                    vy = speed * math.sin(angle)
                    if i % 2 == 0:
                        trial_objects.append(sphere_spec(PRIMARY_OBJECT_ID + i, 0.18, (x, y, 0.18), (vx, vy, 0.0),
                                                         mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color))
                    else:
                        size = pick(DYNAMIC_CUBE_SIZES, rng)
                        trial_objects.append(cube_spec(PRIMARY_OBJECT_ID + i, size, (x, y, size[2] / 2), (vx, vy, 0.0),
                                                       mass=1.0, restitution=0.8, lateral_friction=0.0, color_name=color))
                if not has_initial_overlap(trial_objects, margin=0.04):
                    objects = trial_objects
                    break
                actual_cluster_radius += 0.05
            if not objects:
                raise RuntimeError("Could not place non-overlapping outward S8/L14 objects")
            configs.append(make_cfg("outward_separation", "relative_velocity_separating", "outward_velocity",
                                   (ground_spec(), *objects),
                                   metadata={"all_initial_pairwise_dot_r_v_positive": True, "requested_cluster_radius": cluster_radius, "actual_cluster_radius": actual_cluster_radius}))

    else:
        raise ValueError(f"S8 supports levels 1..14, got L{level_id}")

    return with_ids(configs, level_id, start_id, seed, shuffle=False)


def target_count_for_level(level_id: int, requested_count: int | None) -> int:
    if requested_count is not None:
        return requested_count
    if level_id not in LEVEL_TARGETS:
        raise ValueError(f"S8 supports levels 1..14, got L{level_id}")
    return LEVEL_TARGETS[level_id]


def take_samples(level_id: int, start_id: int, seed: int, count: int | None) -> list[SampleSpec]:
    target_count = target_count_for_level(level_id, count)
    candidate_count = max(target_count * 10, target_count + 50)
    return build_s8_stratified_level_configs(level_id, start_id, seed, candidate_count)


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
    if spec.object_type in {"cube", "ground", "wall", "obstacle"}:
        assert spec.size is not None
        scale = tuple(v / 2.0 for v in spec.size)
        return kb.Cube(scale=scale, **kwargs)
    if spec.object_type == "cylinder":
        assert spec.radius is not None and spec.height is not None
        return kb.Cylinder(scale=(spec.radius, spec.radius, spec.height / 2.0), **kwargs)
    raise ValueError(f"Unsupported object type for S8: {spec.object_type}")


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
    impulse_meta = sample.metadata or {}
    impulse_enabled = bool(impulse_meta.get("external_impulse_enabled"))
    impulse_start_step = int(float(impulse_meta.get("intervention_time", -1.0)) * scene.step_rate)
    impulse_duration_steps = max(1, int(impulse_meta.get("duration_frames", 1)) * steps_per_frame)
    impulse_target = int(impulse_meta.get("target_object", -1))
    impulse_vector = np.array(impulse_meta.get("impulse_vector", (0.0, 0.0, 0.0)), dtype=np.float64)

    for step in range(total_steps):
        if impulse_enabled and impulse_start_step <= step < impulse_start_step + impulse_duration_steps and impulse_target in assets_by_id:
            target_spec = next(spec for spec in sample.objects if spec.object_id == impulse_target)
            target_body = assets_by_id[impulse_target].linked_objects[simulator]
            linear_velocity, angular_velocity = simulator._physics_client.getBaseVelocity(target_body)
            delta_v = impulse_vector / max(float(target_spec.mass or 1.0), 1e-6) / impulse_duration_steps
            simulator._physics_client.resetBaseVelocity(
                target_body,
                linearVelocity=(np.array(linear_velocity, dtype=np.float64) + delta_v).astype(float).tolist(),
                angularVelocity=angular_velocity,
            )
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
    labels = compute_physics_labels(sample, frame_states, force_payloads, FPS)
    if not labels.get("no_dynamic_collision", False):
        shutil.rmtree(view_outputs[0][2], ignore_errors=True)
        raise FilteredNegativeSample(
            f"filtered S{SCENE_ID}/L{sample.level_id} candidate {sample.sample_id}: "
            f"dynamic_dynamic_contact_count={labels.get('total_dynamic_dynamic_contact_count')}"
        )

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
        target_count = target_count_for_level(level_id, args.samples_per_level)
        physical_samples = take_samples(level_id, 1, args.seed, args.samples_per_level)
        output_id = args.start_id
        accepted = 0
        for physical_idx, sample in enumerate(physical_samples, start=1):
            if accepted >= target_count:
                break
            last_output_id = output_id + len(args.views) - 1
            print(
                f"Generating S{SCENE_ID}/L{sample.level_id} physical {physical_idx} "
                f"as video dirs {output_id}-{last_output_id}: {sample.level_name}"
            )
            try:
                generate_physical_sample(args, sample, output_id)
            except FilteredNegativeSample as exc:
                print(exc)
                continue
            accepted += 1
            output_id += len(args.views)
        if accepted < target_count:
            raise RuntimeError(f"Only accepted {accepted}/{target_count} S{SCENE_ID}/L{level_id} samples after filtering")


if __name__ == "__main__":
    main()
