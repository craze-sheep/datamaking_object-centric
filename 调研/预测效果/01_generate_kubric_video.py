#!/usr/bin/env python3
"""Generate a tiny Kubric physics video for SlotFormer probing.

Run this script with Blender's Python. It intentionally only creates the input
video and metadata; SlotFormer inference is handled by 02_slotformer_predict.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import kubric as kb
from kubric.renderer.blender import Blender
from kubric.simulator import PyBullet


def material(color, roughness=0.45):
    return kb.PrincipledBSDFMaterial(color=color, roughness=roughness)


def rgba_to_mp4(rgba, path: Path, fps: int) -> None:
    rgb = rgba[..., :3]
    imageio.mimsave(path, list(rgb), fps=fps, quality=8, macro_block_size=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/home/lzy/project/slot-datamaking/调研/预测效果")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--samples", type=int, default=16)
    argv = None
    # Blender passes its own flags before "--"; keep script args after it.
    import sys
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = output_dir / "kubric_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    scene = kb.Scene(resolution=(args.resolution, args.resolution))
    scene.frame_start = 0
    scene.frame_end = args.num_frames - 1
    scene.frame_rate = args.fps
    scene.step_rate = 240
    scene.gravity = (0.0, 0.0, 0.0)
    scene.background = (0.02, 0.025, 0.035, 1.0)

    scene.camera = kb.OrthographicCamera(
        name="camera",
        position=(0.0, 0.0, 5.0),
        look_at=(0.0, 0.0, 0.0),
        orthographic_scale=2.6,
    )
    scene += kb.DirectionalLight(
        name="sun",
        position=(-2.5, -3.0, 5.0),
        look_at=(0.0, 0.0, 0.0),
        intensity=2.4,
    )

    floor = kb.Cube(
        name="floor",
        scale=(1.35, 1.35, 0.04),
        position=(0.0, 0.0, -0.08),
        static=True,
        background=True,
        material=material((0.72, 0.75, 0.80, 1.0)),
        friction=0.05,
        restitution=0.95,
        segmentation_id=10,
    )
    room_dynamics = dict(static=True, background=True, friction=0.0, restitution=0.98)
    north_wall = kb.Cube(
        name="north_wall", scale=(1.42, 0.05, 0.18), position=(0.0, 1.28, 0.10),
        material=material((0.90, 0.92, 0.95, 1.0)), segmentation_id=11, **room_dynamics)
    south_wall = kb.Cube(
        name="south_wall", scale=(1.42, 0.05, 0.18), position=(0.0, -1.28, 0.10),
        material=material((0.90, 0.92, 0.95, 1.0)), segmentation_id=12, **room_dynamics)
    east_wall = kb.Cube(
        name="east_wall", scale=(0.05, 1.42, 0.18), position=(1.28, 0.0, 0.10),
        material=material((0.90, 0.92, 0.95, 1.0)), segmentation_id=13, **room_dynamics)
    west_wall = kb.Cube(
        name="west_wall", scale=(0.05, 1.42, 0.18), position=(-1.28, 0.0, 0.10),
        material=material((0.90, 0.92, 0.95, 1.0)), segmentation_id=14, **room_dynamics)
    scene += [floor, north_wall, south_wall, east_wall, west_wall]

    object_a = kb.Sphere(
        name="red_ball",
        scale=(0.16, 0.16, 0.16),
        position=(-0.82, -0.10, 0.16),
        velocity=(1.45, 0.18, 0.0),
        mass=0.7,
        friction=0.0,
        restitution=0.92,
        material=material((0.90, 0.08, 0.05, 1.0)),
        segmentation_id=1,
    )
    object_b = kb.Sphere(
        name="yellow_ball",
        scale=(0.20, 0.20, 0.20),
        position=(0.28, 0.10, 0.20),
        velocity=(0.0, 0.0, 0.0),
        mass=1.1,
        friction=0.0,
        restitution=0.82,
        material=material((0.95, 0.76, 0.04, 1.0)),
        segmentation_id=2,
    )
    scene += [object_a, object_b]

    simulator = PyBullet(scene, scratch_dir)
    renderer = Blender(
        scene,
        scratch_dir,
        adaptive_sampling=True,
        use_denoising=True,
        samples_per_pixel=args.samples,
    )

    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end)
    renderer.save_state(output_dir / "kubric_scene.blend")
    frames = renderer.render(return_layers=("rgba", "segmentation"))

    video_path = output_dir / "kubric_input.mp4"
    rgba_to_mp4(frames["rgba"], video_path, fps=args.fps)
    imageio.imwrite(output_dir / "kubric_first_frame.png", frames["rgba"][0, ..., :3])

    summary = {
        "video": str(video_path),
        "num_frames": args.num_frames,
        "fps": args.fps,
        "resolution": [args.resolution, args.resolution],
        "objects": {
            "red_ball": {"initial_position": [-0.82, -0.10, 0.16], "initial_velocity": [1.45, 0.18, 0.0]},
            "yellow_ball": {"initial_position": [0.28, 0.10, 0.20], "initial_velocity": [0.0, 0.0, 0.0]},
        },
        "num_collision_events": len(collisions),
        "sample_collision_events": [
            {
                "frame": float(event["frame"]),
                "instances": [None if x is None else x.name for x in event["instances"]],
                "force": float(event["force"]),
            }
            for event in collisions[:12]
        ],
        "animation_keys": [obj.name for obj in animation.keys()],
    }
    with open(output_dir / "kubric_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote Kubric video: {video_path}")


if __name__ == "__main__":
    main()
