#!/usr/bin/env python3
"""Generate a Kubric video bundle for task1_副本.

Outputs under the video folder:
  frame/                   RGB frame PNGs
  object_segment/          per-frame object binary masks
  label/                   label JSON files, including all object_labels
  rgb.mp4                  RGB video composed from frames
  segmentation_color.mp4   colored instance mask video
  segmentation_ids.npy     raw Kubric instance ids, shape [T, H, W]

Run with Kubric's Blender launcher, for example:
  /root/project/kubric-main/run_with_blender.sh \
    /root/project/task1_副本/output/video/generate_video.py -- \
    --output_dir /root/project/task1_副本/output/video
"""

from __future__ import annotations

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

ID_COLORS = {
    0: (18, 20, 24),
    1: (230, 55, 45),
    2: (245, 205, 55),
    10: (140, 145, 155),
}


def material(name: str, roughness: float = 0.45):
    return kb.PrincipledBSDFMaterial(color=COLORS[name], roughness=roughness)


def make_scene(resolution: int, num_frames: int, seed: int) -> kb.Scene:
    scene = kb.Scene(resolution=(resolution, resolution))
    scene.frame_start = 0
    scene.frame_end = num_frames - 1
    scene.frame_rate = 12
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


def add_collision_setup(scene: kb.Scene):
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
    floor.metadata.update(
        {
            "shape": "cube",
            "mass": None,
            "friction": 0.18,
            "restitution": 0.72,
            "static": True,
        }
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
            "static": False,
        }
    )

    scene += [floor, object_a, object_b]
    return floor, [object_a, object_b]


def segmentation_to_rgb(segmentation_ids: np.ndarray) -> np.ndarray:
    out = np.zeros((*segmentation_ids.shape, 3), dtype=np.uint8)
    for idx, color in ID_COLORS.items():
        out[segmentation_ids == idx] = color
    out[~np.isin(segmentation_ids, list(ID_COLORS))] = (180, 80, 220)
    return out


def save_mp4(frames: np.ndarray, path: Path, fps: int) -> None:
    imageio.mimsave(path, list(frames), fps=fps, quality=8, macro_block_size=1)


def object_label(obj) -> dict:
    label = {
        "name": obj.name,
        "segmentation_id": int(obj.segmentation_id),
    }
    for key, value in obj.metadata.items():
        label[key] = float(value) if isinstance(value, (int, float)) else value
    return label


def collisions_for_objects(collisions: list[dict], tracked_names: list[str]) -> list[dict]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent),
        help="Video output folder. Defaults to this script's folder.",
    )
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    raw_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None
    return parser.parse_args(raw_args)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    if out_dir.exists():
        for child in out_dir.iterdir():
            if child.name == Path(__file__).name:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_dir = out_dir / "frame"
    object_segment_dir = out_dir / "object_segment"
    label_dir = out_dir / "label"
    scratch_dir = out_dir / "scratch"
    for path in (frame_dir, object_segment_dir, label_dir, scratch_dir):
        path.mkdir(parents=True, exist_ok=True)

    scene = make_scene(args.resolution, args.num_frames, args.seed)
    floor, objects = add_collision_setup(scene)

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
    rendered = renderer.render(return_layers=("rgba", "segmentation", "depth"))

    rgb = rendered["rgba"][..., :3].astype(np.uint8)
    segmentation_ids = rendered["segmentation"][..., 0].astype(np.int32)
    depth = rendered["depth"][..., 0].astype(np.float32)
    segmentation_rgb = segmentation_to_rgb(segmentation_ids)

    save_mp4(rgb, out_dir / "rgb.mp4", scene.frame_rate)
    save_mp4(segmentation_rgb, out_dir / "segmentation_color.mp4", scene.frame_rate)
    np.save(out_dir / "segmentation_ids.npy", segmentation_ids)
    np.save(out_dir / "depth.npy", depth)

    all_objects = objects + [floor]
    object_labels = [object_label(obj) for obj in all_objects]
    for frame_idx, frame in enumerate(rgb):
        imageio.imwrite(frame_dir / f"frame_{frame_idx:04d}.png", frame)
        frame_segment_dir = object_segment_dir / f"frame_{frame_idx:04d}"
        frame_segment_dir.mkdir(parents=True, exist_ok=True)
        for obj in all_objects:
            obj_mask = (segmentation_ids[frame_idx] == int(obj.segmentation_id)).astype(np.uint8) * 255
            imageio.imwrite(frame_segment_dir / f"{obj.name}.png", obj_mask)
            np.save(frame_segment_dir / f"{obj.name}.npy", obj_mask.astype(bool))

    labels = {
        "video": str(out_dir / "rgb.mp4"),
        "frame_dir": str(frame_dir),
        "object_segment_dir": str(object_segment_dir),
        "segmentation_video": str(out_dir / "segmentation_color.mp4"),
        "segmentation_ids_npy": str(out_dir / "segmentation_ids.npy"),
        "depth_npy": str(out_dir / "depth.npy"),
        "fps": int(scene.frame_rate),
        "num_frames": int(args.num_frames),
        "duration_seconds": float(args.num_frames / scene.frame_rate),
        "resolution": [int(args.resolution), int(args.resolution)],
        "object_labels": object_labels,
        "collision_events": collisions_for_objects(collisions, [obj.name for obj in objects]),
        "note": "object_labels contains all scene objects used for n:n slot alignment.",
    }
    with open(label_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    with open(label_dir / "object_labels.json", "w", encoding="utf-8") as f:
        json.dump(object_labels, f, indent=2, ensure_ascii=False)

    print(f"Wrote video bundle to: {out_dir}")
    print(f"frames={len(rgb)} fps={scene.frame_rate}")
    print(f"object_labels={[(x['name'], x['segmentation_id']) for x in object_labels]}")
    print(f"collision_events={len(labels['collision_events'])}")
    print(f"animation_frames={len(animation)}")


if __name__ == "__main__":
    main()
