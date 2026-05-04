#!/usr/bin/env python3
"""Audit what Kubric's PyBullet wrapper provides and compare it with task labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - the report still works without numpy.
    np = None


PYBULLET_NATIVE = {
    "animation": {
        "fields": ["position", "quaternion", "velocity", "angular_velocity"],
        "source": "kubric-main/kubric/simulator/pybullet.py:273-309",
        "note": "PyBullet.run() samples these once per rendered frame and returns them per PhysicalObject.",
    },
    "collisions": {
        "fields": ["instances", "position", "contact_normal", "frame", "force"],
        "source": "kubric-main/kubric/simulator/pybullet.py:279-299",
        "note": "These are contact samples from every physics step where normal_force > 1e-6.",
    },
    "physical_object_attributes": {
        "fields": ["mass", "friction", "restitution", "static", "velocity", "angular_velocity", "segmentation_id"],
        "source": "kubric-main/kubric/core/objects.py:160-199",
        "note": "These are Kubric PhysicalObject attributes used by the PyBullet backend.",
    },
}


KUBRIC_OR_RENDERER_HELPERS = {
    "instance_info": {
        "fields": [
            "positions",
            "quaternions",
            "velocities",
            "angular_velocities",
            "mass",
            "friction",
            "restitution",
            "image_positions",
            "bboxes_3d",
        ],
        "source": "kubric-main/kubric/utils.py:159-184",
        "note": "Convenience collector after simulation keyframes exist; bboxes_3d are computed from object geometry and pose.",
    },
    "scene_metadata": {
        "fields": ["resolution", "frame_rate", "step_rate", "gravity", "num_frames"],
        "source": "kubric-main/kubric/utils.py:132-142",
        "note": "Scene-level metadata, not per-object physics state.",
    },
    "visibility_and_2d_bbox": {
        "fields": ["visibility_pixel_count", "bboxes", "bbox_frames"],
        "source": "kubric-main/kubric/post_processing.py:21-68",
        "note": "Computed from rendered segmentation; not PyBullet-native.",
    },
    "renderer_layers": {
        "fields": ["rgba", "segmentation", "backward_flow", "forward_flow", "depth", "normal", "object_coordinates"],
        "source": "kubric-main/kubric/renderer/blender.py:140-150,275-298",
        "note": "Blender render outputs; optical flow and depth live here, not in PyBullet.",
    },
}


AUDIT_ROWS = [
    {
        "name": "Mass / Friction / Restitution",
        "user_claim": "原生直接拿",
        "verdict": "正确",
        "reason": "它们是 Kubric PhysicalObject 属性，也会被写入 PyBullet body 参数。",
        "recommended_storage": "object_properties.mass/friction/restitution",
    },
    {
        "name": "Static",
        "user_claim": "配置项记录",
        "verdict": "正确",
        "reason": "static 是 PhysicalObject 属性，决定物体是否参与动态仿真。",
        "recommended_storage": "object_properties.static",
    },
    {
        "name": "Shape / Material / Color",
        "user_claim": "配置项记录",
        "verdict": "基本正确",
        "reason": "shape/material/color 是 Kubric/Blender 场景配置，不是 PyBullet 运动输出；要手动写入 metadata。",
        "recommended_storage": "object_properties.shape/material/color",
    },
    {
        "name": "Position / Quaternion",
        "user_claim": "原生直接拿",
        "verdict": "正确",
        "reason": "PyBullet.getBasePositionAndOrientation() 被 Kubric 包装成 animation[obj]['position'/'quaternion']。",
        "recommended_storage": "object_states.positions/quaternions",
    },
    {
        "name": "Linear Velocity",
        "user_claim": "原生直接拿",
        "verdict": "正确",
        "reason": "PyBullet.getBaseVelocity() 第一项就是线速度。",
        "recommended_storage": "object_states.velocities",
    },
    {
        "name": "Angular Velocity",
        "user_claim": "原生直接拿",
        "verdict": "正确",
        "reason": "PyBullet.getBaseVelocity() 第二项就是刚体角速度；它不是 2D 轨迹转向代理。",
        "recommended_storage": "object_states.angular_velocities",
    },
    {
        "name": "Visibility",
        "user_claim": "原生直接拿",
        "verdict": "需要修正",
        "reason": "Kubric 可以从 segmentation 后处理得到可见像素数，但 PyBullet 本身不提供 visibility。",
        "recommended_storage": "derived_from_render.visibility 或 visibility_ratio",
    },
    {
        "name": "Depth",
        "user_claim": "标准输出",
        "verdict": "正确但不是 PyBullet",
        "reason": "depth 是 Blender renderer layer，不是物理仿真器输出。",
        "recommended_storage": "render_layers.depth",
    },
    {
        "name": "Optical Flow",
        "user_claim": "标准输出",
        "verdict": "正确但当前示例未保存",
        "reason": "Kubric Blender renderer 支持 forward_flow/backward_flow；当前快照对应的生成脚本只请求 rgba/segmentation/depth。",
        "recommended_storage": "render_layers.forward_flow/backward_flow",
    },
    {
        "name": "3D BBox",
        "user_claim": "标准输出",
        "verdict": "需要修正",
        "reason": "Kubric 可以用 instance.bbox_3d/get_instance_info 逐帧计算；不是 renderer 默认数组，也不是 PyBullet contact 输出。",
        "recommended_storage": "object_states.bboxes_3d",
    },
    {
        "name": "2D BBox",
        "user_claim": "未单独列为选中项",
        "verdict": "派生",
        "reason": "由 segmentation mask 后处理得到。",
        "recommended_storage": "derived_from_segmentation.bboxes_2d",
    },
    {
        "name": "Acceleration",
        "user_claim": "简单计算",
        "verdict": "正确",
        "reason": "由相邻帧 velocity 差分得到，推荐用 3D velocity 而不是 2D mask 中心。",
        "recommended_storage": "derived_motion.accelerations",
    },
    {
        "name": "Speed / Angular Speed",
        "user_claim": "简单计算",
        "verdict": "正确",
        "reason": "分别是 velocity 和 angular_velocity 的模长。",
        "recommended_storage": "derived_motion.speed/angular_speed",
    },
    {
        "name": "Collision",
        "user_claim": "原生直接拿",
        "verdict": "一半正确",
        "reason": "PyBullet 原生给 contact samples；如果标注 Collision event，应把连续 contact segment 的起点视为碰撞瞬时事件。",
        "recommended_storage": "pair_events.collision_events",
    },
    {
        "name": "Contact",
        "user_claim": "简单推导",
        "verdict": "正确",
        "reason": "由 collision/contact samples 按 pair 聚合成持续区间。",
        "recommended_storage": "pair_events.contact_intervals",
    },
    {
        "name": "Strong Collision",
        "user_claim": "简单推导",
        "verdict": "正确",
        "reason": "由 contact force 阈值判断。",
        "recommended_storage": "pair_events.strong_collision_events",
    },
    {
        "name": "Stop / Start Moving / Sliding / Falling / Spinning",
        "user_claim": "阈值或多条件判定",
        "verdict": "正确",
        "reason": "这些都是从速度、角速度、接触状态派生的 object-level events。",
        "recommended_storage": "object_events.*",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def npy_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    if np is None:
        return {"exists": True, "shape": None, "dtype": None, "note": "numpy not available"}
    arr = np.load(path)
    return {"exists": True, "shape": list(arr.shape), "dtype": str(arr.dtype)}


def existing_or_fallback(raw_path: Any, fallback_path: Path) -> Path:
    if raw_path:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate
    return fallback_path


def inspect_current_outputs(labels_json: Path, alignment_json: Path | None) -> dict[str, Any]:
    labels = load_json(labels_json)
    label_dir = labels_json.parent
    video_dir = label_dir.parent
    events = labels.get("collision_events", [])
    event_pairs = Counter(tuple(e.get("instances", [])) for e in events)

    present_object_fields = sorted({key for item in labels.get("object_labels", []) for key in item.keys()})
    present_label_fields = sorted(labels.keys())
    missing_native_motion = ["positions", "quaternions", "velocities", "angular_velocities", "bboxes_3d"]
    segmentation_path = existing_or_fallback(
        labels.get("segmentation_ids_npy"),
        video_dir / "segmentation_ids.npy",
    )
    depth_path = existing_or_fallback(
        labels.get("depth_npy"),
        video_dir / "depth.npy",
    )

    alignment = None
    if alignment_json and alignment_json.exists():
        alignment = load_json(alignment_json)

    return {
        "labels_json": str(labels_json),
        "label_top_level_fields": present_label_fields,
        "object_label_fields": present_object_fields,
        "num_objects": len(labels.get("object_labels", [])),
        "num_collision_samples": len(events),
        "collision_event_fields": sorted({key for item in events for key in item.keys()}),
        "collision_pairs": {str(list(pair)): count for pair, count in event_pairs.items()},
        "saved_arrays": {
            "segmentation_ids": {
                **npy_info(segmentation_path),
                "checked_path": str(segmentation_path),
                "label_path": labels.get("segmentation_ids_npy"),
            },
            "depth": {
                **npy_info(depth_path),
                "checked_path": str(depth_path),
                "label_path": labels.get("depth_npy"),
            },
            "slot_masks": npy_info(video_dir.parent / "slot" / "slot_masks.npy"),
            "slots": npy_info(video_dir.parent / "slot" / "slots.npy"),
        },
        "slot_alignment": {
            "alignment_json": str(alignment_json) if alignment_json else None,
            "available": alignment is not None,
            "object_to_slot": None if alignment is None else alignment.get("object_to_slot"),
            "mean_iou": None if alignment is None else alignment.get("mean_iou"),
        },
        "missing_from_current_labels_but_pybullet_or_kubric_can_provide": missing_native_motion,
    }


def write_report(output_md: Path, audit_json: Path, payload: dict[str, Any]) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    inv = payload["current_output_inventory"]
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("# task2: Kubric PyBullet 数据能力核对\n\n")
        f.write("## 结论\n\n")
        f.write("- Kubric 的 `PyBullet.run()` **直接返回**两类数据：每帧物体运动状态 `animation`，以及物理步级别的接触采样 `collisions`。\n")
        f.write("- 你写的 `Mass/Friction/Restitution/Static/Position/Quaternion/Linear Velocity/Angular Velocity` 基本判断正确，属于 Kubric/PyBullet 可直接拿。\n")
        f.write("- `Visibility/Depth/Optical Flow/2D BBox` 不是 PyBullet 输出；它们来自 Blender 渲染层或 segmentation 后处理。\n")
        f.write("- `3D BBox` 是 Kubric 根据物体几何和每帧 pose 计算，不能算 PyBullet 原生 contact 输出。\n")
        f.write("- `Collision` 原始数据可直接拿，但那是很多 contact samples；标注事件时要聚合连续 contact 区间。\n\n")

        f.write("## PyBullet 直接提供\n\n")
        for name, item in PYBULLET_NATIVE.items():
            f.write(f"- `{name}`: {', '.join(item['fields'])}\n")
            f.write(f"  - 来源: `{item['source']}`\n")
            f.write(f"  - 说明: {item['note']}\n")

        f.write("\n## Kubric/Renderer 可提供，但不是 PyBullet 原生\n\n")
        for name, item in KUBRIC_OR_RENDERER_HELPERS.items():
            f.write(f"- `{name}`: {', '.join(item['fields'])}\n")
            f.write(f"  - 来源: `{item['source']}`\n")
            f.write(f"  - 说明: {item['note']}\n")

        f.write("\n## 你的表格逐项核对\n\n")
        f.write("| 项目 | 你的判断 | 核对结果 | 原因 | 建议保存字段 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for row in AUDIT_ROWS:
            f.write(
                f"| {row['name']} | {row['user_claim']} | **{row['verdict']}** | "
                f"{row['reason']} | `{row['recommended_storage']}` |\n"
            )

        f.write("\n## 当前 task2 输入快照实际包含什么\n\n")
        f.write(f"- labels: `{inv['labels_json']}`\n")
        f.write(f"- top-level 字段: `{', '.join(inv['label_top_level_fields'])}`\n")
        f.write(f"- object 字段: `{', '.join(inv['object_label_fields'])}`\n")
        f.write(f"- object 数量: `{inv['num_objects']}`\n")
        f.write(f"- collision/contact samples 数量: `{inv['num_collision_samples']}`\n")
        f.write(f"- collision sample 字段: `{', '.join(inv['collision_event_fields'])}`\n")
        f.write("- collision pair 计数:\n")
        for pair, count in inv["collision_pairs"].items():
            f.write(f"  - `{pair}`: `{count}`\n")
        f.write("- 数组文件:\n")
        for key, info in inv["saved_arrays"].items():
            f.write(f"  - `{key}`: `{info}`\n")
        f.write(f"- slot 对齐可用: `{inv['slot_alignment']['available']}`\n")
        f.write(f"- object_to_slot: `{inv['slot_alignment']['object_to_slot']}`\n")

        f.write("\n## 需要修正你原表述的地方\n\n")
        f.write("1. `Visibility` 不应写成 PyBullet 原生直接拿；应写成 Kubric segmentation 后处理。\n")
        f.write("2. `Depth` 和 `Optical Flow` 是 Blender renderer layer，不是 PyBullet。\n")
        f.write("3. `3D BBox` 是 Kubric 几何+pose 计算；可以直接用 `get_instance_info` 拿，但来源不是 PyBullet contact。\n")
        f.write("4. `Collision` 可以从 PyBullet contact samples 直接拿原始数据，但事件标签需要聚合，否则一次持续接触会被记成很多次碰撞。\n")
        f.write("5. 当前 task2 输入快照的 `labels.json` 还没有保存 `position/quaternion/velocity/angular_velocity/bboxes_3d`，下一步应补到生成脚本。\n")

        f.write("\n## 推荐 task2 数据结构\n\n")
        f.write("```jsonc\n")
        f.write("""{
  "object_properties": {
    "mass": "质量；Kubric/PyBullet 物体属性，静态背景可为 null",
    "friction": "摩擦系数；影响滑动和停止",
    "restitution": "弹性/反弹系数；影响碰撞后的反弹",
    "static": "是否静态物体；用于过滤 floor/background",
    "shape": "几何形状；例如 sphere/cube",
    "material": "材质名称或材质参数；需要生成脚本写入 metadata",
    "color": "颜色标签；用于视觉属性和 slot 对齐分析",
    "segmentation_id": "Kubric 实例分割 ID；连接 mask 和 object",
    "slot_id": "task1 匈牙利匹配得到的 slot 编号"
  },
  "object_states": {
    "positions": "每帧 3D 位置 [x, y, z]；PyBullet 直接提供",
    "quaternions": "每帧 3D 姿态四元数 [w, x, y, z]；PyBullet 直接提供",
    "velocities": "每帧线速度 [vx, vy, vz]；PyBullet 直接提供",
    "angular_velocities": "每帧角速度 [wx, wy, wz]；PyBullet 直接提供，表示真实自转",
    "bboxes_3d": "每帧 3D 包围盒 8 个角点；Kubric 由几何和 pose 计算"
  },
  "render_layers": {
    "depth": "深度图；Blender renderer 输出，不是 PyBullet",
    "segmentation": "实例分割图；每个像素对应 object id",
    "forward_flow": "前向光流；像素从当前帧到下一帧的位移",
    "backward_flow": "后向光流；像素从当前帧到上一帧的位移"
  },
  "derived_motion": {
    "accelerations": "加速度；由相邻帧 velocities 差分得到",
    "speed": "线速度模长；用于 stop/start/sliding 阈值",
    "angular_speed": "角速度模长；用于 spinning/sliding 阈值"
  },
  "derived_segmentation": {
    "visibility_ratio": "可见比例；由每帧可见像素数归一化得到",
    "bboxes_2d": "2D 外接框；由 segmentation mask 计算"
  },
  "pair_events": {
    "contact_intervals": "接触区间；由 PyBullet contact samples 按 pair 聚合",
    "collision_events": "碰撞事件；通常取连续 contact 区间的起始帧",
    "strong_collision_events": "强碰撞；collision force 超过阈值的碰撞事件"
  },
  "object_events": {
    "start_moving": "开始运动；speed 从低于阈值变为高于阈值",
    "stop": "停止；speed 和 angular_speed 都低于阈值",
    "sliding": "滑动；有接触、线速度较大、角速度较小",
    "falling": "坠落；z 方向向下运动且没有接触",
    "spinning": "自转；angular_speed 高且 speed 低"
  }
}""")
        f.write("\n```\n")


def parse_args() -> argparse.Namespace:
    task2_dir = Path(__file__).resolve().parents[1]
    default_labels = task2_dir / "input" / "video" / "label" / "labels.json"
    default_alignment = task2_dir / "input" / "alignment" / "alignment_result.json"
    default_out = task2_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_json", default=str(default_labels))
    parser.add_argument("--alignment_json", default=str(default_alignment))
    parser.add_argument("--output_dir", default=str(default_out))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_json = Path(args.labels_json).resolve()
    alignment_json = Path(args.alignment_json).resolve() if args.alignment_json else None
    output_dir = Path(args.output_dir).resolve()

    payload = {
        "pybullet_native": PYBULLET_NATIVE,
        "kubric_or_renderer_helpers": KUBRIC_OR_RENDERER_HELPERS,
        "audit_rows": AUDIT_ROWS,
        "current_output_inventory": inspect_current_outputs(labels_json, alignment_json),
    }
    write_report(
        output_md=output_dir / "提取结果.md",
        audit_json=output_dir / "output" / "pybullet_data_audit.json",
        payload=payload,
    )
    print(f"Wrote {output_dir / '提取结果.md'}")
    print(f"Wrote {output_dir / 'output' / 'pybullet_data_audit.json'}")


if __name__ == "__main__":
    main()
