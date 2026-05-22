"""Generate a small Kubric showcase dataset with labels and MP4 videos.

This is intentionally small and clean: a few short clips for group-meeting demos.
It uses primitive Kubric objects, PyBullet dynamics, and Blender rendering.
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
    "green": (0.12, 0.55, 0.20, 1.0),
    "blue": (0.08, 0.20, 0.85, 1.0),
    "yellow": (0.95, 0.80, 0.08, 1.0),
    "floor": (0.78, 0.80, 0.84, 1.0),
    "ramp": (0.55, 0.58, 0.62, 1.0),
}

ID_COLORS = {
    0: (18, 20, 24),
    1: (230, 55, 45),
    2: (245, 205, 55),
    10: (150, 155, 165),
    11: (95, 130, 220),
}


def material(name, roughness=0.45):
    return kb.PrincipledBSDFMaterial(color=COLORS[name], roughness=roughness)


def add_camera_and_light(scene):
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


def make_scene(resolution, num_frames):
    scene = kb.Scene(resolution=(resolution, resolution))
    scene.frame_start = 0
    scene.frame_end = num_frames - 1
    scene.frame_rate = 12
    scene.step_rate = 240
    scene.gravity = (0.0, 0.0, -9.81)
    scene.background = (0.04, 0.05, 0.06, 1.0)
    add_camera_and_light(scene)
    return scene


def add_floor(scene, friction=0.55, restitution=0.25):
    floor = kb.Cube(
        name="floor",
        scale=(3.4, 3.4, 0.08),
        position=(0.0, 0.0, -0.08),
        static=True,
        material=material("floor"),
        friction=friction,
        restitution=restitution,
        segmentation_id=10,
        background=True,
    )
    scene += floor
    return floor


def moving_object(name, shape, color, mass, friction, restitution, deformability,
                  position, velocity=(0, 0, 0), scale=0.22):
    kwargs = dict(
        name=name,
        scale=scale,
        position=position,
        velocity=velocity,
        mass=mass,
        friction=friction,
        restitution=restitution,
        material=material(color),
        segmentation_id=1 if name == "object_a" else 2,
    )
    obj = kb.Cube(**kwargs) if shape == "cube" else kb.Sphere(**kwargs)
    obj.metadata.update({
        "shape": shape,
        "mass": mass,
        "friction": friction,
        "restitution": restitution,
        # Kubric/PyBullet primitive soft-body simulation is not used here.
        # This is a supervised label/proxy for the first-stage dataset.
        "deformability": deformability,
    })
    return obj


def build_free_fall(scene):
    add_floor(scene, friction=0.35, restitution=0.85)
    obj = moving_object(
        "object_a", "sphere", "red",
        mass=0.7, friction=0.25, restitution=0.88, deformability=0.0,
        position=(-0.35, 0.0, 2.15),
        velocity=(0.9, 0.0, 0.0),
    )
    scene += obj
    return {
        "scenario": "free_fall_bounce",
        "target_event": "bounce",
        "objects": [obj],
    }


def build_ramp_slide(scene):
    add_floor(scene, friction=0.70, restitution=0.15)
    ramp = kb.Cube(
        name="inclined_plane",
        scale=(1.65, 0.72, 0.07),
        position=(-0.2, 0.0, 0.72),
        euler=(0.0, 0.42, 0.0),
        static=True,
        material=material("ramp"),
        friction=0.18,
        restitution=0.05,
        segmentation_id=11,
        background=True,
    )
    scene += ramp
    obj = moving_object(
        "object_a", "cube", "green",
        mass=1.4, friction=0.12, restitution=0.12, deformability=0.0,
        position=(-0.95, 0.0, 1.55),
        velocity=(0.0, 0.0, 0.0),
        scale=0.18,
    )
    scene += obj
    return {
        "scenario": "inclined_slide",
        "target_event": "slide",
        "objects": [obj],
    }


def build_collision(scene):
    add_floor(scene, friction=0.18, restitution=0.72)
    obj_a = moving_object(
        "object_a", "sphere", "blue",
        mass=0.55, friction=0.15, restitution=0.82, deformability=0.0,
        position=(-1.25, 0.0, 0.26),
        velocity=(2.6, 0.0, 0.0),
        scale=0.22,
    )
    obj_b = moving_object(
        "object_b", "sphere", "yellow",
        mass=1.25, friction=0.30, restitution=0.42, deformability=0.35,
        position=(0.35, 0.0, 0.26),
        velocity=(0.0, 0.0, 0.0),
        scale=0.24,
    )
    scene += [obj_a, obj_b]
    return {
        "scenario": "collision_transfer",
        "target_event": "collision",
        "objects": [obj_a, obj_b],
    }


SCENARIOS = {
    "free_fall": build_free_fall,
    "ramp_slide": build_ramp_slide,
    "collision": build_collision,
}


def rgba_to_mp4(rgba, path, fps):
    rgb = rgba[..., :3]
    imageio.mimsave(path, list(rgb), fps=fps, quality=8, macro_block_size=1)


def segmentation_to_rgb(segmentation):
    seg = segmentation[..., 0].astype(np.int32)
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for idx, color in ID_COLORS.items():
        out[seg == idx] = color
    unknown = ~np.isin(seg, list(ID_COLORS))
    out[unknown] = (180, 80, 220)

    # Add a subtle grid/ground contrast so the video reads well in slides.
    yy, xx = np.indices(seg.shape[-2:])
    grid = ((xx // 24 + yy // 24) % 2).astype(np.uint8)
    bg = seg == 0
    out[bg & (grid == 1)] = (24, 27, 32)
    return out


def segmentation_to_mp4(segmentation, path, fps):
    rgb = segmentation_to_rgb(segmentation)
    imageio.mimsave(path, list(rgb), fps=fps, quality=8, macro_block_size=1)


def collision_summary(collisions, tracked):
    tracked_names = {obj: obj.name for obj in tracked}
    tracked_name_set = set(tracked_names.values())
    events = []
    for event in collisions:
        a, b = event["instances"]
        names = [None if a is None else a.name, None if b is None else b.name]
        if names[0] in tracked_name_set or names[1] in tracked_name_set:
            events.append({
                "frame": float(event["frame"]),
                "instances": names,
                "force": float(event["force"]),
                "position": [float(x) for x in event["position"]],
            })
    return events


def summarize_events(collisions, tracked):
    events = collision_summary(collisions, tracked)
    tracked_name_set = {obj.name for obj in tracked}
    object_object = [
        e for e in events
        if e["instances"][0] in tracked_name_set and e["instances"][1] in tracked_name_set
    ]
    floor_contacts = [
        e for e in events
        if "floor" in e["instances"] and any(name in tracked_name_set for name in e["instances"])
    ]
    return {
        "num_contacts_total": len(events),
        "num_object_object_contacts": len(object_object),
        "first_object_object_contact": object_object[0] if object_object else None,
        "first_floor_contact": floor_contacts[0] if floor_contacts else None,
        "sample_contacts": events[:20],
    }


def render_case(name, builder, output_dir, resolution, num_frames, samples):
    case_dir = output_dir / name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = case_dir / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    scene = make_scene(resolution, num_frames)
    spec = builder(scene)
    simulator = PyBullet(scene, scratch_dir)
    renderer = Blender(
        scene,
        scratch_dir,
        adaptive_sampling=True,
        use_denoising=True,
        samples_per_pixel=samples,
    )

    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end)
    renderer.save_state(case_dir / "scene.blend")
    frames = renderer.render(return_layers=("rgba", "segmentation", "depth"))

    rgba_to_mp4(frames["rgba"], case_dir / "rgb.mp4", fps=scene.frame_rate)
    segmentation_to_mp4(frames["segmentation"], case_dir / "presentation.mp4",
                        fps=scene.frame_rate)
    segmentation_to_mp4(frames["segmentation"], case_dir / "segmentation_color.mp4",
                        fps=scene.frame_rate)
    kb.write_image_dict({"rgba": frames["rgba"][:6],
                         "segmentation": frames["segmentation"][:6]},
                        case_dir / "preview_frames")
    imageio.imwrite(case_dir / "presentation_preview.png",
                    segmentation_to_rgb(frames["segmentation"])[min(8, len(frames["segmentation"]) - 1)])

    event_summary = summarize_events(collisions, spec["objects"])
    metadata = {
        "video": str(case_dir / "rgb.mp4"),
        "presentation_video": str(case_dir / "presentation.mp4"),
        "segmentation_video": str(case_dir / "segmentation_color.mp4"),
        "scenario": spec["scenario"],
        "target_event": spec["target_event"],
        "num_frames": num_frames,
        "fps": scene.frame_rate,
        "resolution": [resolution, resolution],
        "property_labels": {
            obj.name: dict(obj.metadata) for obj in spec["objects"]
        },
        "events": event_summary,
        "event_labels": {
            "has_collision": event_summary["num_contacts_total"] > 0,
            "has_object_object_collision": event_summary["num_object_object_contacts"] > 0,
            "event_type": spec["target_event"],
        },
        "note": "deformability is a first-stage supervised/proxy label; PyBullet primitive soft-body deformation is not enabled.",
    }
    with open(case_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output/mini_physics_showcase")
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument("--num_frames", type=int, default=36)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metadata = []
    for scenario in args.scenarios:
        all_metadata.append(render_case(
            scenario,
            SCENARIOS[scenario],
            output_dir,
            args.resolution,
            args.num_frames,
            args.samples,
        ))
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"Wrote showcase dataset to {output_dir}")


if __name__ == "__main__":
    main()
