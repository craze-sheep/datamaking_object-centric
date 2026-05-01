#!/usr/bin/env python3
"""Extract selected physical quantities and events from Kubric outputs.

Selected quantities:
- Mass, Friction, Restitution, Static
- Linear Velocity, Angular Velocity (2D trajectory turn-rate proxy)
- Visibility
- Acceleration
- Speed, Angular Speed

Selected events:
- Collision, Contact, Strong Collision
- Stop, Start Moving, Sliding, Falling, Spinning
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_json", required=True)
    parser.add_argument("--segmentation_ids_npy", required=True)
    parser.add_argument("--alignment_result_json", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--strong_collision_quantile", type=float, default=0.9)
    parser.add_argument("--speed_start_threshold", type=float, default=1.0)
    parser.add_argument("--speed_stop_threshold", type=float, default=0.7)
    parser.add_argument("--sliding_ang_speed_max", type=float, default=1.4)
    parser.add_argument("--falling_vy_threshold", type=float, default=1.0)
    parser.add_argument("--spinning_speed_max", type=float, default=1.2)
    parser.add_argument("--spinning_ang_speed_min", type=float, default=2.8)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_serializable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def contiguous_segments(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    out = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        out.append([start, prev])
        start = idx
        prev = idx
    out.append([start, prev])
    return out


def rising_edges(mask: np.ndarray, include_zero: bool = True) -> list[int]:
    frames = []
    if include_zero and bool(mask[0]):
        frames.append(0)
    for t in range(1, len(mask)):
        if bool(mask[t]) and not bool(mask[t - 1]):
            frames.append(t)
    return frames


def get_centroid(mask_2d: np.ndarray) -> tuple[float | None, float | None]:
    ys, xs = np.where(mask_2d)
    if len(xs) == 0:
        return None, None
    return float(xs.mean()), float(ys.mean())


def compute_motion_features(positions_xy: np.ndarray, fps: float) -> dict:
    num_frames = positions_xy.shape[0]
    pos = positions_xy.copy()

    # Fill missing values with previous valid value to keep derivatives stable.
    for t in range(num_frames):
        if np.isnan(pos[t]).any():
            if t == 0:
                later = np.where(~np.isnan(pos[:, 0]))[0]
                if len(later) > 0:
                    pos[t] = pos[later[0]]
                else:
                    pos[t] = np.array([0.0, 0.0], dtype=np.float64)
            else:
                pos[t] = pos[t - 1]

    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * fps

    acc = np.zeros_like(pos)
    acc[1:] = (vel[1:] - vel[:-1]) * fps

    speed = np.linalg.norm(vel, axis=1)

    # Angular velocity proxy: heading change rate of trajectory (not rigid-body self-spin).
    theta = np.arctan2(vel[:, 1], vel[:, 0])
    omega = np.zeros(num_frames, dtype=np.float64)
    for t in range(1, num_frames):
        if speed[t] < 1e-8 or speed[t - 1] < 1e-8:
            omega[t] = 0.0
            continue
        dtheta = (theta[t] - theta[t - 1] + np.pi) % (2 * np.pi) - np.pi
        omega[t] = dtheta * fps
    ang_speed = np.abs(omega)

    return {
        "position_xy": pos,
        "linear_velocity_xy": vel,
        "acceleration_xy": acc,
        "speed": speed,
        "angular_velocity_proxy": omega,
        "angular_speed_proxy": ang_speed,
    }


def main() -> None:
    args = parse_args()

    labels_path = Path(args.labels_json).resolve()
    seg_path = Path(args.segmentation_ids_npy).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_json(labels_path)
    seg = np.load(seg_path).astype(np.int32)

    fps = float(labels.get("fps", 12))
    num_frames = int(labels.get("num_frames", seg.shape[0]))
    frame_ids = list(range(num_frames))

    object_labels = labels.get("object_labels", [])
    id_to_meta = {int(x["segmentation_id"]): x for x in object_labels}
    name_to_id = {str(x["name"]): int(x["segmentation_id"]) for x in object_labels}
    id_to_name = {v: k for k, v in name_to_id.items()}

    slot_mapping = {}
    if args.alignment_result_json:
        alignment = load_json(Path(args.alignment_result_json).resolve())
        slot_mapping = {int(k): int(v) for k, v in alignment.get("slot_to_object", {}).items()}
    object_to_slot = {obj_id: slot for slot, obj_id in slot_mapping.items()}

    objects_out = {}
    static_name_set = {
        str(x.get("name"))
        for x in object_labels
        if bool(x.get("static", False))
    }

    for obj_id, meta in id_to_meta.items():
        name = str(meta.get("name", f"id_{obj_id}"))
        px = (seg == obj_id).sum(axis=(1, 2)).astype(np.float64)
        max_px = float(px.max()) if float(px.max()) > 0 else 1.0
        visibility = (px / max_px).astype(np.float64)

        pos = np.full((num_frames, 2), np.nan, dtype=np.float64)
        for t in range(num_frames):
            cx, cy = get_centroid(seg[t] == obj_id)
            if cx is None:
                continue
            pos[t, 0] = cx
            pos[t, 1] = cy

        motion = compute_motion_features(pos, fps)

        objects_out[name] = {
            "name": name,
            "segmentation_id": obj_id,
            "slot_id": object_to_slot.get(obj_id),
            "shape": meta.get("shape"),
            "mass": meta.get("mass"),
            "friction": meta.get("friction"),
            "restitution": meta.get("restitution"),
            "static": bool(meta.get("static", False)),
            "frames": frame_ids,
            "visibility": visibility,
            "position_xy": motion["position_xy"],
            "linear_velocity_xy": motion["linear_velocity_xy"],
            "acceleration_xy": motion["acceleration_xy"],
            "speed": motion["speed"],
            "angular_velocity_proxy": motion["angular_velocity_proxy"],
            "angular_speed_proxy": motion["angular_speed_proxy"],
        }

    # Pair-level contact/collision extraction from collision_events.
    pair_frame_force = defaultdict(dict)  # pair_key -> frame -> max_force
    pair_frame_pos = defaultdict(dict)

    for e in labels.get("collision_events", []):
        a, b = e.get("instances", [None, None])
        if a is None or b is None:
            continue
        pair = tuple(sorted((str(a), str(b))))
        f_raw = float(e.get("frame", 0.0))
        f_idx = int(np.clip(np.rint(f_raw), 0, num_frames - 1))
        force = float(e.get("force", 0.0))

        prev = pair_frame_force[pair].get(f_idx, -np.inf)
        if force >= prev:
            pair_frame_force[pair][f_idx] = force
            pair_frame_pos[pair][f_idx] = [float(x) for x in e.get("position", [0.0, 0.0, 0.0])]

    pair_events_out = {}
    dynamic_collision_forces = []
    all_collision_forces = []

    for pair, frame_to_force in pair_frame_force.items():
        frames_sorted = sorted(frame_to_force.keys())
        contact_segments = contiguous_segments(frames_sorted)
        collision_frames = [segm[0] for segm in contact_segments]

        collision_items = []
        for f_idx in collision_frames:
            item = {
                "frame": f_idx,
                "time_sec": float(f_idx / fps),
                "force": float(frame_to_force[f_idx]),
                "position": pair_frame_pos[pair][f_idx],
            }
            collision_items.append(item)
            all_collision_forces.append(float(frame_to_force[f_idx]))

            a, b = pair
            if a not in static_name_set and b not in static_name_set:
                dynamic_collision_forces.append(float(frame_to_force[f_idx]))

        pair_events_out[f"{pair[0]}|{pair[1]}"] = {
            "pair": [pair[0], pair[1]],
            "pair_segmentation_ids": [name_to_id.get(pair[0]), name_to_id.get(pair[1])],
            "contact_frames": frames_sorted,
            "contact_intervals": contact_segments,
            "collision_events": collision_items,
            "strong_collision_events": [],
        }

    force_base = dynamic_collision_forces if dynamic_collision_forces else all_collision_forces
    if force_base:
        strong_threshold = float(np.quantile(np.array(force_base, dtype=np.float64), args.strong_collision_quantile))
    else:
        strong_threshold = float("inf")

    for key, item in pair_events_out.items():
        a, b = item["pair"]
        is_dynamic_pair = (a not in static_name_set) and (b not in static_name_set)
        if not is_dynamic_pair:
            continue
        strong_events = [
            e
            for e in item["collision_events"]
            if float(e["force"]) >= strong_threshold
        ]
        item["strong_collision_events"] = strong_events

    # Object-level events derived from per-frame quantities.
    for obj_name, item in objects_out.items():
        speed = np.asarray(item["speed"], dtype=np.float64)
        ang_speed = np.asarray(item["angular_speed_proxy"], dtype=np.float64)
        vy = np.asarray(item["linear_velocity_xy"], dtype=np.float64)[:, 1]
        is_static = bool(item.get("static", False))

        contact_any = np.zeros(num_frames, dtype=bool)
        contact_with_static = np.zeros(num_frames, dtype=bool)

        for pair_key, pair_item in pair_events_out.items():
            n1, n2 = pair_item["pair"]
            if obj_name not in (n1, n2):
                continue
            other = n2 if n1 == obj_name else n1
            cframes = pair_item["contact_frames"]
            contact_any[cframes] = True
            if other in static_name_set:
                contact_with_static[cframes] = True

        moving = speed >= float(args.speed_start_threshold)
        stationary = (speed <= float(args.speed_stop_threshold)) & (ang_speed <= float(args.sliding_ang_speed_max))

        if is_static:
            start_moving_frames = []
            stop_frames = []
            sliding_mask = np.zeros(num_frames, dtype=bool)
            falling_mask = np.zeros(num_frames, dtype=bool)
            spinning_mask = np.zeros(num_frames, dtype=bool)
        else:
            start_moving_frames = rising_edges(moving, include_zero=False)
            stop_frames = [
                t
                for t in range(1, num_frames)
                if bool(stationary[t]) and bool(moving[t - 1])
            ]
            sliding_mask = moving & (ang_speed <= float(args.sliding_ang_speed_max)) & contact_with_static
            falling_mask = (vy >= float(args.falling_vy_threshold)) & (~contact_any)
            spinning_mask = (speed <= float(args.spinning_speed_max)) & (
                ang_speed >= float(args.spinning_ang_speed_min)
            )

        item["object_events"] = {
            "contact_any_frames": np.where(contact_any)[0].tolist(),
            "start_moving_frames": start_moving_frames,
            "stop_frames": stop_frames,
            "sliding_frames": np.where(sliding_mask)[0].tolist(),
            "sliding_intervals": contiguous_segments(np.where(sliding_mask)[0].tolist()),
            "falling_frames": np.where(falling_mask)[0].tolist(),
            "falling_intervals": contiguous_segments(np.where(falling_mask)[0].tolist()),
            "spinning_frames": np.where(spinning_mask)[0].tolist(),
            "spinning_intervals": contiguous_segments(np.where(spinning_mask)[0].tolist()),
        }

    out = {
        "meta": {
            "labels_json": str(labels_path),
            "segmentation_ids_npy": str(seg_path),
            "fps": fps,
            "num_frames": num_frames,
            "resolution": labels.get("resolution"),
            "slot_mapping_used": bool(object_to_slot),
            "notes": {
                "angular_velocity_proxy": "2D trajectory heading-rate proxy, not true rigid-body spin.",
                "visibility": "pixel_count(frame)/max_pixel_count(object)",
            },
            "thresholds": {
                "strong_collision_quantile": args.strong_collision_quantile,
                "strong_collision_force_threshold": strong_threshold,
                "speed_start_threshold": args.speed_start_threshold,
                "speed_stop_threshold": args.speed_stop_threshold,
                "sliding_ang_speed_max": args.sliding_ang_speed_max,
                "falling_vy_threshold": args.falling_vy_threshold,
                "spinning_speed_max": args.spinning_speed_max,
                "spinning_ang_speed_min": args.spinning_ang_speed_min,
            },
        },
        "selected_quantities": [
            "mass",
            "friction",
            "restitution",
            "static",
            "linear_velocity_xy",
            "angular_velocity_proxy",
            "visibility",
            "acceleration_xy",
            "speed",
            "angular_speed_proxy",
        ],
        "selected_events": [
            "collision",
            "contact",
            "strong_collision",
            "stop",
            "start_moving",
            "sliding",
            "falling",
            "spinning",
        ],
        "objects": objects_out,
        "pair_events": pair_events_out,
    }

    out_json = out_dir / "physics_events.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=to_serializable)

    # Light summary markdown.
    summary_md = out_dir / "summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# 物理量与事件提取结果\n\n")
        f.write(f"- labels: `{labels_path}`\n")
        f.write(f"- seg ids: `{seg_path}`\n")
        f.write(f"- 输出 JSON: `{out_json}`\n")
        f.write(f"- 帧数/FPS: `{num_frames}` / `{fps}`\n")
        f.write(f"- Strong Collision 阈值: `{strong_threshold:.6f}`\n\n")

        f.write("## 物体级\n")
        for obj_name, item in objects_out.items():
            ev = item["object_events"]
            f.write(f"- {obj_name} (id={item['segmentation_id']}, slot={item['slot_id']}):\n")
            f.write(f"  - Start Moving: {ev['start_moving_frames']}\n")
            f.write(f"  - Stop: {ev['stop_frames']}\n")
            f.write(f"  - Sliding 区间: {ev['sliding_intervals']}\n")
            f.write(f"  - Falling 区间: {ev['falling_intervals']}\n")
            f.write(f"  - Spinning 区间: {ev['spinning_intervals']}\n")

        f.write("\n## 双体级\n")
        for key, item in pair_events_out.items():
            f.write(f"- {key}:\n")
            f.write(f"  - Contact 区间: {item['contact_intervals']}\n")
            f.write(
                "  - Collision 帧: "
                + str([int(e["frame"]) for e in item["collision_events"]])
                + "\n"
            )
            f.write(
                "  - Strong Collision 帧: "
                + str([int(e["frame"]) for e in item["strong_collision_events"]])
                + "\n"
            )

    print(f"Wrote: {out_json}")
    print(f"Wrote: {summary_md}")


if __name__ == "__main__":
    main()
