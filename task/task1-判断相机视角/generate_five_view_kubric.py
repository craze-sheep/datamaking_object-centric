#!/usr/bin/env python3
"""Generate one synchronized Kubric sample rendered from five camera views.

The camera definitions follow task/数据集构成.md:
front, back, left, right, and top. The physics simulation is executed once;
then the same keyed animation is rendered from each camera.
"""

from __future__ import annotations

import argparse
import gc
import json
import numbers
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import kubric as kb
from kubric.renderer.blender import Blender
from kubric.simulator import PyBullet


VIEW_SPECS = {
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
    "left": {
        "type": "Perspective",
        "position": (-7.5, 0.0, 3.2),
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
}

COLORS = {
    "floor": (0.72, 0.74, 0.78, 1.0),
    "red": (0.88, 0.10, 0.08, 1.0),
    "blue": (0.08, 0.20, 0.85, 1.0),
    "green": (0.12, 0.55, 0.20, 1.0),
    "yellow": (0.95, 0.80, 0.08, 1.0),
    "cyan": (0.05, 0.70, 0.80, 1.0),
    "purple": (0.55, 0.20, 0.75, 1.0),
    "wall": (0.58, 0.60, 0.64, 1.0),
}


def parse_args() -> argparse.Namespace:
    default_output_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=str(default_output_dir),
        help="All generated files are written under this directory.",
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--samples", type=int, default=32)

    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    return parser.parse_args(argv)


def material(color_id: str, roughness: float = 0.55, metallic: float = 0.0):
    return kb.PrincipledBSDFMaterial(
        color=COLORS[color_id],
        roughness=roughness,
        metallic=metallic,
    )


def rgba_to_mp4(rgba, path: Path, fps: int) -> None:
    rgb = rgba[..., :3]
    imageio.mimsave(path, list(rgb), fps=fps, quality=8, macro_block_size=1)


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
    scene.unlink_view(renderer)
    for asset in scene.assets:
        asset.linked_objects.pop(renderer, None)


def make_camera(view_id: str, spec: dict):
    if spec["type"] == "Perspective":
        return kb.PerspectiveCamera(
            name=f"camera_{view_id}",
            position=spec["position"],
            look_at=spec["look_at"],
            focal_length=spec["focal_length"],
            sensor_width=spec["sensor_width"],
        )
    if spec["type"] == "Orthographic":
        return kb.OrthographicCamera(
            name=f"camera_{view_id}",
            position=spec["position"],
            look_at=spec["look_at"],
            orthographic_scale=spec["orthographic_scale"],
        )
    raise ValueError(f"Unsupported camera type: {spec['type']}")


def build_scene(resolution: int, num_frames: int, fps: int):
    scene = kb.Scene(resolution=(resolution, resolution))
    scene.frame_start = 0
    scene.frame_end = num_frames - 1
    scene.frame_rate = fps
    scene.step_rate = 240
    scene.gravity = (0.0, 0.0, -9.81)
    scene.background = (0.035, 0.04, 0.048, 1.0)
    scene.ambient_illumination = (0.08, 0.08, 0.08, 1.0)

    cameras = {view_id: make_camera(view_id, spec) for view_id, spec in VIEW_SPECS.items()}
    scene += list(cameras.values())
    scene.camera = cameras["front"]

    scene += kb.DirectionalLight(
        name="sun",
        position=(-3.0, -4.0, 7.0),
        look_at=(0.0, 0.0, 0.0),
        intensity=2.8,
    )

    floor = kb.Cube(
        name="floor",
        scale=(4.0, 4.0, 0.08),
        position=(0.0, 0.0, -0.08),
        static=True,
        background=True,
        material=material("floor"),
        friction=0.3,
        restitution=0.5,
        segmentation_id=10,
    )
    wall_x = kb.Cube(
        name="wall_x_plus",
        scale=(0.08, 1.8, 0.7),
        position=(1.4, 0.0, 0.6),
        static=True,
        background=True,
        material=material("wall"),
        friction=0.2,
        restitution=0.65,
        segmentation_id=11,
    )
    y_marker = kb.Cube(
        name="green_y_marker_column",
        scale=(0.16, 0.16, 0.8),
        position=(0.1, 0.55, 0.45),
        static=True,
        background=True,
        material=material("green"),
        friction=0.2,
        restitution=0.2,
        segmentation_id=12,
    )
    scene += [floor, wall_x, y_marker]

    active_ball = kb.Sphere(
        name="active_red_sphere_m",
        scale=(0.22, 0.22, 0.22),
        position=(-1.3, -0.18, 0.26),
        velocity=(2.5, 0.0, 0.0),
        mass=1.0,
        friction=0.20,
        restitution=0.85,
        material=material("red", roughness=0.35),
        segmentation_id=1,
    )
    target_ball = kb.Sphere(
        name="target_blue_sphere_m",
        scale=(0.22, 0.22, 0.22),
        position=(0.35, 0.0, 0.26),
        velocity=(0.0, 0.0, 0.0),
        mass=1.0,
        friction=0.20,
        restitution=0.85,
        material=material("blue", roughness=0.35),
        segmentation_id=2,
    )
    yellow_x_marker = kb.Cube(
        name="yellow_negative_x_marker",
        scale=(0.16, 0.16, 0.16),
        position=(-1.55, 1.25, 0.16),
        static=True,
        background=True,
        material=material("yellow"),
        friction=0.3,
        restitution=0.2,
        segmentation_id=13,
    )
    scene += [active_ball, target_ball, yellow_x_marker]

    sample_spec = {
        "scenario": "S5_ball_hits_ball_with_orientation_markers",
        "shared_simulation": True,
        "dynamic_objects": {
            "active_red_sphere_m": {
                "shape": "sphere_m",
                "initial_position": [-1.3, -0.18, 0.26],
                "initial_velocity": [2.5, 0.0, 0.0],
                "mass": 1.0,
                "friction": 0.20,
                "restitution": 0.85,
            },
            "target_blue_sphere_m": {
                "shape": "sphere_m",
                "initial_position": [0.35, 0.0, 0.26],
                "initial_velocity": [0.0, 0.0, 0.0],
                "mass": 1.0,
                "friction": 0.20,
                "restitution": 0.85,
            },
        },
        "static_orientation_markers": {
            "wall_x_plus": {"position": [1.4, 0.0, 0.6], "meaning": "+x side wall"},
            "green_y_marker_column": {"position": [0.1, 0.55, 0.45], "meaning": "+y marker"},
            "yellow_negative_x_marker": {
                "position": [-1.55, 1.25, 0.16],
                "meaning": "-x/+y corner marker",
            },
        },
    }
    return scene, cameras, sample_spec


def collision_summary(collisions):
    events = []
    for event in collisions[:40]:
        a, b = event["instances"]
        events.append(
            {
                "frame": float(event["frame"]),
                "instances": [None if x is None else x.name for x in (a, b)],
                "force": float(event["force"]),
                "position": [float(v) for v in event["position"]],
            }
        )
    return events


def json_default(value):
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views_dir = output_dir / "views"
    scratch_root = output_dir / "scratch"
    if views_dir.exists():
        shutil.rmtree(views_dir)
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    views_dir.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)

    scene, cameras, sample_spec = build_scene(args.resolution, args.num_frames, args.fps)
    simulator = PyBullet(scene, scratch_root / "pybullet")
    state_renderer = Blender(
        scene,
        scratch_root / "blender_state",
        adaptive_sampling=True,
        use_denoising=True,
        samples_per_pixel=args.samples,
    )

    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end)
    state_renderer.save_state(output_dir / "five_view_scene.blend")
    unlink_renderer(scene, state_renderer)
    gc.collect()

    view_outputs = {}
    for view_id, camera in cameras.items():
        view_dir = views_dir / view_id
        try:
            view_dir.mkdir(parents=True, exist_ok=True)
            renderer = Blender(
                scene,
                scratch_root / f"blender_{view_id}",
                adaptive_sampling=True,
                use_denoising=True,
                samples_per_pixel=args.samples,
            )
            try:
                replay_scene_keyframes(scene)
                scene.camera = camera
                frames = renderer.render(return_layers=("rgba", "segmentation"))
            finally:
                unlink_renderer(scene, renderer)
                gc.collect()
        except Exception:
            shutil.rmtree(view_dir, ignore_errors=True)
            raise

        video_path = view_dir / f"{view_id}.mp4"
        first_frame_path = view_dir / f"{view_id}_first_frame.png"
        rgba_to_mp4(frames["rgba"], video_path, fps=args.fps)
        imageio.imwrite(first_frame_path, frames["rgba"][0, ..., :3])
        kb.write_palette_png(
            frames["segmentation"][0],
            view_dir / f"{view_id}_segmentation_first_frame.png",
        )

        view_outputs[view_id] = {
            "video": str(video_path),
            "first_frame": str(first_frame_path),
            "segmentation_first_frame": str(view_dir / f"{view_id}_segmentation_first_frame.png"),
            "camera": {
                "type": VIEW_SPECS[view_id]["type"],
                "position": list(VIEW_SPECS[view_id]["position"]),
                "look_at": list(VIEW_SPECS[view_id]["look_at"]),
                **(
                    {
                        "focal_length": VIEW_SPECS[view_id]["focal_length"],
                        "sensor_width": VIEW_SPECS[view_id]["sensor_width"],
                    }
                    if VIEW_SPECS[view_id]["type"] == "Perspective"
                    else {"orthographic_scale": VIEW_SPECS[view_id]["orthographic_scale"]}
                ),
            },
        }

    manifest = {
        "source_spec": "/home/lzy/project/slot-datamaking/task/数据集构成.md",
        "output_dir": str(output_dir),
        "resolution": [args.resolution, args.resolution],
        "num_frames": args.num_frames,
        "fps": args.fps,
        "step_rate": scene.step_rate,
        "gravity": list(scene.gravity),
        "scene_center": [0.0, 0.0, 0.0],
        "sample": sample_spec,
        "views": view_outputs,
        "num_collision_events": len(collisions),
        "sample_collision_events": collision_summary(collisions),
        "animated_assets": [obj.name for obj in animation.keys()],
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=json_default)

    print(f"Wrote five synchronized view videos under: {views_dir}")


if __name__ == "__main__":
    main()
