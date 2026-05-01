#!/usr/bin/env python3
"""Generate a 3-second Kubric collision clip with instance masks.

Run with Kubric's Blender launcher, for example:
  /root/project/kubric-main/run_with_blender.sh /root/project/task/脚本/generate_kubric_video.py -- \
    --output_dir /root/project/task/output/kubric_case
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import kubric as kb
import numpy as np
from kubric.renderer.blender import Blender
from kubric.simulator import PyBullet


COLORS = {
    "red": (0.88, 0.10, 0.08, 1.0),
    "yellow": (0.95, 0.80, 0.08, 1.0),
    "floor": (0.78, 0.80, 0.84, 1.0),
}

# Segmentation id -> visualization color (RGB)
ID_COLORS = {
    0: (18, 20, 24),
    1: (230, 55, 45),    # object_a
    2: (245, 205, 55),   # object_b
    10: (140, 145, 155),  # floor
}


def material(name, roughness=0.45):
    return kb.PrincipledBSDFMaterial(color=COLORS[name], roughness=roughness)


def make_scene(resolution: int, num_frames: int, seed: int):
    scene = kb.Scene(resolution=(resolution, resolution))
    scene.frame_start = 0
    scene.frame_end = num_frames - 1
    scene.frame_rate = 12  # 36 frames / 12 FPS = 3.0 seconds
    scene.step_rate = 240
    scene.gravity = (0.0, 0.0, -9.81)
    scene.background = (0.04, 0.05, 0.06, 1.0)
    scene.metadata["seed"] = seed

    scene.camera = kb.PerspectiveCamera(
        name="camera",
        position=(3.4, -5.2, 3.0),
        look_at=(0.0, 0.0, 0.55),
        focal_length=32,
    )
    scene += kb.DirectionalLight(
        name="sun",
        position=(-3.0, -4.0, 6.0),
        look_at=(0.0, 0.0, 0.0),
        intensity=2.5,
    )
    return scene


def add_collision_setup(scene):
    floor = kb.Cube(
        name="floor",
        scale=(3.4, 3.4, 0.08),
        position=(0.0, 0.0, -0.08),
        static=True,
        material=material("floor"),
        friction=0.18,
        restitution=0.72,
        segmentation_id=10,
        background=True,
    )

    object_a = kb.Sphere(
        name="object_a",
        scale=0.22,
        position=(-1.25, 0.0, 0.26),
        velocity=(2.6, 0.0, 0.0),
        mass=0.55,
        friction=0.15,
        restitution=0.82,
        material=material("red"),
        segmentation_id=1,
    )
    object_a.metadata.update(
        {
            "shape": "sphere",
            "mass": 0.55,
            "friction": 0.15,
            "restitution": 0.82,
            "deformability": 0.0,
            "static": False,
        }
    )

    object_b = kb.Sphere(
        name="object_b",
        scale=0.24,
        position=(0.35, 0.0, 0.26),
        velocity=(0.0, 0.0, 0.0),
        mass=1.25,
        friction=0.30,
        restitution=0.42,
        material=material("yellow"),
        segmentation_id=2,
    )
    object_b.metadata.update(
        {
            "shape": "sphere",
            "mass": 1.25,
            "friction": 0.30,
            "restitution": 0.42,
            "deformability": 0.35,
            "static": False,
        }
    )

    scene += [floor, object_a, object_b]
    return floor, object_a, object_b


def segmentation_to_rgb(segmentation):
    seg = segmentation[..., 0].astype(np.int32)
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for idx, color in ID_COLORS.items():
        out[seg == idx] = color
    unknown = ~np.isin(seg, list(ID_COLORS))
    out[unknown] = (180, 80, 220)
    return out


def save_mp4(frames: np.ndarray, path: Path, fps: int):
    imageio.mimsave(path, list(frames), fps=fps, quality=8, macro_block_size=1)


def collisions_for_objects(collisions, tracked_names):
    events = []
    tracked = set(tracked_names)
    for event in collisions:
        a, b = event["instances"]
        names = [None if a is None else a.name, None if b is None else b.name]
        if names[0] in tracked or names[1] in tracked:
            events.append(
                {
                    "frame": float(event["frame"]),
                    "instances": names,
                    "force": float(event["force"]),
                    "position": [float(x) for x in event["position"]],
                }
            )
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    out_dir = Path(args.output_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_dir / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    scene = make_scene(args.resolution, args.num_frames, args.seed)
    floor, object_a, object_b = add_collision_setup(scene)

    simulator = PyBullet(scene, scratch_dir)
    renderer = Blender(
        scene,
        scratch_dir,
        adaptive_sampling=True,
        use_denoising=True,
        samples_per_pixel=args.samples,
    )

    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end)
    renderer.save_state(out_dir / "scene.blend")
    frames = renderer.render(return_layers=("rgba", "segmentation", "depth"))

    rgb = frames["rgba"][..., :3]
    seg_ids = frames["segmentation"][..., 0].astype(np.int32)
    seg_rgb = segmentation_to_rgb(frames["segmentation"])
    depth = frames["depth"][..., 0].astype(np.float32)

    save_mp4(rgb, out_dir / "rgb.mp4", scene.frame_rate)
    save_mp4(seg_rgb, out_dir / "segmentation_color.mp4", scene.frame_rate)
    np.save(out_dir / "segmentation_ids.npy", seg_ids)
    np.save(out_dir / "depth.npy", depth)

    object_labels = [
        {
            "name": object_a.name,
            "segmentation_id": int(object_a.segmentation_id),
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in object_a.metadata.items()},
        },
        {
            "name": object_b.name,
            "segmentation_id": int(object_b.segmentation_id),
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in object_b.metadata.items()},
        },
        {
            "name": floor.name,
            "segmentation_id": int(floor.segmentation_id),
            "shape": "cube",
            "static": True,
            "mass": None,
            "friction": float(floor.friction),
            "restitution": float(floor.restitution),
        },
    ]

    collision_events = collisions_for_objects(collisions, tracked_names=["object_a", "object_b"])
    labels = {
        "video": str(out_dir / "rgb.mp4"),
        "segmentation_video": str(out_dir / "segmentation_color.mp4"),
        "segmentation_ids_npy": str(out_dir / "segmentation_ids.npy"),
        "depth_npy": str(out_dir / "depth.npy"),
        "fps": int(scene.frame_rate),
        "num_frames": int(args.num_frames),
        "duration_seconds": float(args.num_frames / scene.frame_rate),
        "resolution": [int(args.resolution), int(args.resolution)],
        "object_labels": object_labels,
        "collision_events": collision_events,
        "note": "segmentation_ids.npy stores per-pixel Kubric instance IDs.",
    }
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"Wrote Kubric case to: {out_dir}")
    print(f"duration_seconds={labels['duration_seconds']}")
    print(f"object_labels={[(x['name'], x['segmentation_id']) for x in object_labels]}")
    print(f"collision_events={len(collision_events)}")
    print(f"animation_frames={len(animation)}")


if __name__ == "__main__":
    main()
