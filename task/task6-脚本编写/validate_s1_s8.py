#!/usr/bin/env python3
"""Validate S1-S8 dataset layout, metadata, depth/segmentation, and views."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DB = Path("/home/lzy/project/slot-datamaking/database")
REPORT_PATH = Path("validation_report_s1_s8.txt")

SCENES = [f"S{i}" for i in range(1, 9)]
FRAMES = 36
RESOLUTION = (128, 128)
FPS = 12
DURATION_S = 3.0
GRAVITY = [0.0, 0.0, -9.8]
VIEW_COUNT = {"S1": 2, "S2": 3, "S3": 3, "S4": 2, "S5": 5, "S6": 5, "S7": 5, "S8": 5}
EXPECTED_VIEWS = {
    "S1": {"front", "top"},
    "S2": {"front", "top", "left"},
    "S3": {"front", "top", "left"},
    "S4": {"front", "top"},
    "S5": {"front", "back", "left", "right", "top"},
    "S6": {"front", "back", "left", "right", "top"},
    "S7": {"front", "back", "left", "right", "top"},
    "S8": {"front", "back", "left", "right", "top"},
}
PHYSICAL_LEVEL_TARGETS = {
    "S1": {1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 150, 7: 150},
    "S2": {1: 80, 2: 80, 3: 80, 4: 80, 5: 120, 6: 120, 7: 120},
    "S3": {1: 80, 2: 80, 3: 80, 4: 80, 5: 80, 6: 120, 7: 120, 8: 120, 9: 120},
    "S4": {1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 150, 7: 150, 8: 150, 9: 150},
    "S5": {1: 80, 2: 80, 3: 80, 4: 80, 5: 80, 6: 120, 7: 120, 8: 120, 9: 120},
    "S6": {1: 80, 2: 80, 3: 80, 4: 80, 5: 80, 6: 120, 7: 120, 8: 120, 9: 120},
    "S7": {1: 80, 2: 80, 3: 80, 4: 80, 5: 80, 6: 120, 7: 120, 8: 120, 9: 120, 10: 120, 11: 120, 12: 120},
    "S8": {1: 120, 2: 120, 3: 120, 4: 120, 5: 120, 6: 120, 7: 120, 8: 120, 9: 120, 10: 120, 11: 120, 12: 120, 13: 120, 14: 120},
}

STATIC_COMPARE_KEYS = [
    "object_id",
    "segmentation_id",
    "static",
    "shape",
    "object_type",
    "mass",
    "radius",
    "height",
    "size",
    "lateralFriction",
    "rollingFriction",
    "spinningFriction",
    "restitution",
    "color",
    "color_name",
    "rgba",
    "position",
    "velocity",
    "angular_velocity",
]
DYNAMIC_COMPARE_KEYS = ["position", "velocity", "angular_velocity"]
CHECK_FRAME_NUMS = [1, FRAMES // 2, FRAMES]


errors: list[str] = []
warnings: list[str] = []
stats: dict[str, Any] = {}
report_lines: list[str] = []


def emit(msg: str = "") -> None:
    print(msg, flush=True)
    report_lines.append(msg)


def err(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


def read_json(path: Path, label: str) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        err(f"{label}: JSON解析失败: {exc}")
        return None


def list_level_dirs(scene_dir: Path) -> dict[int, Path]:
    levels: dict[int, Path] = {}
    for child in scene_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("L") and child.name[1:].isdigit():
            levels[int(child.name[1:])] = child
        else:
            warn(f"{scene_dir.name}: 跳过非level目录 {child.name}")
    return levels


def list_sample_ids(level_dir: Path, scene: str, level: int) -> list[int]:
    ids: list[int] = []
    for child in level_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.isdigit():
            ids.append(int(child.name))
        else:
            warn(f"{scene}/L{level}: 跳过非数字样本目录 {child.name}")
    return sorted(ids)


def expected_sample_count(scene: str, level: int) -> int:
    return PHYSICAL_LEVEL_TARGETS[scene][level] * VIEW_COUNT[scene]


def compact_values(values: list[int], limit: int = 30) -> str:
    if len(values) <= limit:
        return str(values)
    return f"{values[:limit]} ... 共{len(values)}个"


def same_value(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_value(x, y, tol) for x, y in zip(a, b))
    return a == b


def static_by_id(rows: Any, label: str) -> dict[int, dict[str, Any]]:
    if not isinstance(rows, list):
        err(f"{label}: object_static.json 顶层不是list")
        return {}
    by_id: dict[int, dict[str, Any]] = {}
    seen: set[int] = set()
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            err(f"{label}: object_static[{idx}] 不是object")
            continue
        object_id = row.get("object_id")
        segmentation_id = row.get("segmentation_id")
        if not isinstance(object_id, int):
            err(f"{label}: object_static[{idx}].object_id 非整数: {object_id!r}")
            continue
        if object_id in seen:
            err(f"{label}: object_id重复: {object_id}")
        seen.add(object_id)
        if segmentation_id != object_id:
            err(f"{label}: object_id={object_id} segmentation_id不一致: {segmentation_id!r}")
        by_id[object_id] = row
    return by_id


def load_static(level_dir: Path, sid: int, scene: str, level: int) -> dict[int, dict[str, Any]]:
    path = level_dir / str(sid) / "object_static.json"
    data = read_json(path, f"{scene}/L{level}/{sid}/object_static.json")
    return static_by_id(data, f"{scene}/L{level}/{sid}") if data is not None else {}


def sample_indices(n: int) -> list[int]:
    candidates = set(range(min(5, n)))
    candidates.update(range(max(0, n // 2 - 2), min(n, n // 2 + 3)))
    candidates.update(range(max(0, n - 5), n))
    return sorted(candidates)


def event_indices(event_count: int) -> list[int]:
    if event_count <= 20:
        return list(range(event_count))
    picks = set(range(5))
    picks.update(range(max(0, event_count // 2 - 5), min(event_count, event_count // 2 + 5)))
    picks.update(range(max(0, event_count - 5), event_count))
    return sorted(picks)


def validate_depth_npz(path: Path, label: str) -> None:
    try:
        with np.load(path) as data:
            if "depth" not in data.files:
                err(f"{label}: npz缺少depth key, keys={data.files}")
                return
            depth = data["depth"]
            if depth.shape != (FRAMES, *RESOLUTION):
                err(f"{label}: depth shape错误 {depth.shape}, 期望 {(FRAMES, *RESOLUTION)}")
            if np.all(depth == 0):
                warn(f"{label}: depth全零")
    except Exception as exc:
        err(f"{label}: npz读取失败: {exc}")


def validate_segment_npz(path: Path, label: str, expected_object_id: int, expected_frame: int) -> bool:
    try:
        with np.load(path) as data:
            keys = set(data.files)
            required = {"mask", "object_id", "segmentation_id", "frame_num"}
            missing = sorted(required - keys)
            if missing:
                err(f"{label}: segment npz缺少key {missing}, keys={sorted(keys)}")
                return False
            mask = data["mask"]
            object_id = int(np.asarray(data["object_id"]).item())
            segmentation_id = int(np.asarray(data["segmentation_id"]).item())
            frame_num = int(np.asarray(data["frame_num"]).item())
            if mask.shape != RESOLUTION:
                err(f"{label}: mask shape错误 {mask.shape}, 期望 {RESOLUTION}")
            if not np.issubdtype(mask.dtype, np.integer) and mask.dtype != np.bool_:
                err(f"{label}: mask dtype错误 {mask.dtype}, 期望整数或bool")
            if object_id != expected_object_id:
                err(f"{label}: object_id={object_id}, 期望 {expected_object_id}")
            if segmentation_id != expected_object_id:
                err(f"{label}: segmentation_id={segmentation_id}, 期望 {expected_object_id}")
            if frame_num != expected_frame:
                err(f"{label}: frame_num={frame_num}, 期望 {expected_frame}")
            return bool(np.any(mask))
    except Exception as exc:
        err(f"{label}: segment npz读取失败: {exc}")
        return False


def validate_video_json(path: Path, label: str, scene: str, sid: int) -> str | None:
    data = read_json(path, label)
    if data is None:
        return None
    if not isinstance(data, dict):
        err(f"{label}: 顶层不是object")
        return None
    if data.get("fps") != FPS:
        err(f"{label}: fps={data.get('fps')}, 期望 {FPS}")
    if data.get("duration_s") != DURATION_S:
        err(f"{label}: duration_s={data.get('duration_s')}, 期望 {DURATION_S}")
    if data.get("num_frames") != FRAMES:
        err(f"{label}: num_frames={data.get('num_frames')}, 期望 {FRAMES}")
    if data.get("gravity") != GRAVITY:
        err(f"{label}: gravity={data.get('gravity')}, 期望 {GRAVITY}")
    cameras = data.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != 1:
        err(f"{label}: cameras数量={len(cameras) if isinstance(cameras, list) else '非list'}, 期望单camera")
        return None
    cam = cameras[0]
    if not isinstance(cam, dict):
        err(f"{label}: camera不是object")
        return None
    view_name = cam.get("view_name")
    if view_name not in EXPECTED_VIEWS[scene]:
        err(f"{label}: view_name={view_name!r}, 不在预期集合 {sorted(EXPECTED_VIEWS[scene])}")
    if cam.get("resolution") != list(RESOLUTION):
        err(f"{label}: resolution={cam.get('resolution')}, 期望 {list(RESOLUTION)}")
    if cam.get("video_path") != f"{sid}.mp4":
        err(f"{label}: video_path={cam.get('video_path')!r}, 期望 {sid}.mp4")
    if cam.get("depth_path") != f"{sid}.npz":
        err(f"{label}: depth_path={cam.get('depth_path')!r}, 期望 {sid}.npz")
    return view_name if isinstance(view_name, str) else None


def validate_force_matrix(path: Path, label: str, object_ids: list[int]) -> None:
    data = read_json(path, label)
    if data is None:
        return
    if not isinstance(data, dict):
        err(f"{label}: 顶层不是object")
        return
    order = data.get("object_order")
    if order != object_ids:
        err(f"{label}: object_order={order}, 期望 {object_ids}")
    matrix = data.get("force_matrix")
    if not isinstance(matrix, list):
        err(f"{label}: force_matrix不是list")


scene_levels: dict[str, dict[int, Path]] = {}
level_samples: dict[tuple[str, int], list[int]] = {}
static_cache: dict[tuple[str, int, int], dict[int, dict[str, Any]]] = {}
view_cache: dict[tuple[str, int, int], str | None] = {}


emit("=" * 70)
emit("S1-S8 全面数据验证")
emit("=" * 70)


# ========== Phase 1: 文件完整性 ==========
emit("\n--- Phase 1: 文件完整性 ---")
for scene in SCENES:
    scene_dir = DB / scene
    if not scene_dir.is_dir():
        err(f"{scene}: 场景目录不存在")
        continue
    levels = list_level_dirs(scene_dir)
    scene_levels[scene] = levels
    expected_levels = set(PHYSICAL_LEVEL_TARGETS[scene])
    actual_levels = set(levels)
    missing_levels = sorted(expected_levels - actual_levels)
    extra_levels = sorted(actual_levels - expected_levels)
    if missing_levels:
        err(f"{scene}: 缺失level {['L' + str(x) for x in missing_levels]}")
    if extra_levels:
        err(f"{scene}: 多余level {['L' + str(x) for x in extra_levels]}")

    for level in sorted(expected_levels & actual_levels):
        level_dir = levels[level]
        sample_ids = list_sample_ids(level_dir, scene, level)
        level_samples[(scene, level)] = sample_ids
        expected_count = expected_sample_count(scene, level)
        expected_ids = set(range(1, expected_count + 1))
        actual_ids = set(sample_ids)
        missing_ids = sorted(expected_ids - actual_ids)
        extra_ids = sorted(actual_ids - expected_ids)
        if missing_ids:
            err(f"{scene}/L{level}: 样本序号缺失 {compact_values(missing_ids)}")
        if extra_ids:
            err(f"{scene}/L{level}: 样本序号多余 {compact_values(extra_ids)}")
        if len(sample_ids) != expected_count:
            err(f"{scene}/L{level}: 样本数={len(sample_ids)}, 期望 {expected_count}")
        if len(sample_ids) % VIEW_COUNT[scene] != 0:
            err(f"{scene}/L{level}: 样本数 {len(sample_ids)} 不是视角数 {VIEW_COUNT[scene]} 的倍数")

        missing_files = Counter()
        bad_frames = 0
        bad_frame_dirs = 0
        for sid in sample_ids:
            sample_dir = level_dir / str(sid)
            required_files = [
                sample_dir / "video.json",
                sample_dir / "object_static.json",
                sample_dir / f"{sid}.mp4",
                sample_dir / f"{sid}.npz",
            ]
            for path in required_files:
                if not path.is_file():
                    missing_files[path.name] += 1
                    err(f"{scene}/L{level}/{sid}: 缺少文件 {path.name}")
            dyn_dir = sample_dir / "dynamic"
            if not dyn_dir.is_dir():
                err(f"{scene}/L{level}/{sid}: 缺少dynamic/")
                continue
            frame_ids = sorted(int(d.name) for d in dyn_dir.iterdir() if d.is_dir() and d.name.isdigit())
            expected_frames = list(range(1, FRAMES + 1))
            if frame_ids != expected_frames:
                bad_frames += 1
                missing = sorted(set(expected_frames) - set(frame_ids))
                extra = sorted(set(frame_ids) - set(expected_frames))
                err(f"{scene}/L{level}/{sid}: 帧号异常 missing={missing} extra={extra}")
            for frame in expected_frames:
                frame_dir = dyn_dir / str(frame)
                if not frame_dir.is_dir():
                    continue
                for sub in ["object_dynamicjson", "object_segment"]:
                    if not (frame_dir / sub).is_dir():
                        bad_frame_dirs += 1
                        err(f"{scene}/L{level}/{sid}/dynamic/{frame}: 缺少{sub}/")
                png_path = frame_dir / f"{frame}.png"
                if not png_path.is_file():
                    bad_frame_dirs += 1
                    err(f"{scene}/L{level}/{sid}/dynamic/{frame}: 缺少{frame}.png")
                if not (frame_dir / "force_matrix.json").is_file():
                    bad_frame_dirs += 1
                    err(f"{scene}/L{level}/{sid}/dynamic/{frame}: 缺少force_matrix.json")
        stats[f"{scene}/L{level}_count"] = len(sample_ids)
        emit(
            f"  {scene}/L{level}: 样本={len(sample_ids)}/{expected_count}, "
            f"缺顶层文件={sum(missing_files.values())}, 帧号异常样本={bad_frames}, 帧文件/目录问题={bad_frame_dirs}"
        )


# ========== Phase 2: JSON结构 ==========
emit("\n--- Phase 2: JSON结构 ---")
for scene in SCENES:
    for level in sorted(PHYSICAL_LEVEL_TARGETS[scene]):
        sample_ids = level_samples.get((scene, level), [])
        if not sample_ids:
            continue
        level_dir = scene_levels[scene][level]
        view_names: list[str] = []
        object_counts: list[int] = []
        force_checks = 0
        for sid in sample_ids:
            sample_dir = level_dir / str(sid)
            view_name = validate_video_json(sample_dir / "video.json", f"{scene}/L{level}/{sid}/video.json", scene, sid)
            view_cache[(scene, level, sid)] = view_name
            if view_name:
                view_names.append(view_name)

            static_data = read_json(sample_dir / "object_static.json", f"{scene}/L{level}/{sid}/object_static.json")
            by_id = static_by_id(static_data, f"{scene}/L{level}/{sid}") if static_data is not None else {}
            static_cache[(scene, level, sid)] = by_id
            if by_id:
                object_counts.append(len(by_id))

            object_ids = sorted(by_id)
            for frame in CHECK_FRAME_NUMS:
                fm_path = sample_dir / "dynamic" / str(frame) / "force_matrix.json"
                if fm_path.is_file() and object_ids:
                    validate_force_matrix(fm_path, f"{scene}/L{level}/{sid}/dynamic/{frame}/force_matrix.json", object_ids)
                    force_checks += 1

        stats[f"{scene}/L{level}_views"] = dict(Counter(view_names))
        emit(
            f"  {scene}/L{level}: 视角={Counter(view_names).most_common()}, "
            f"物体数={sorted(set(object_counts))}, force_matrix抽检={force_checks}"
        )


# ========== Phase 3: 深度分割 ==========
emit("\n--- Phase 3: 深度分割 ---")
for scene in SCENES:
    for level in sorted(PHYSICAL_LEVEL_TARGETS[scene]):
        sample_ids = level_samples.get((scene, level), [])
        if not sample_ids:
            continue
        level_dir = scene_levels[scene][level]
        depth_checked = 0
        segment_checked = 0
        segment_zero = 0
        missing_segment = 0
        missing_dynamic_json = 0

        for sid in sample_ids:
            sample_dir = level_dir / str(sid)
            depth_path = sample_dir / f"{sid}.npz"
            if depth_path.is_file():
                validate_depth_npz(depth_path, f"{scene}/L{level}/{sid}/{sid}.npz")
                depth_checked += 1

            by_id = static_cache.get((scene, level, sid))
            if by_id is None:
                by_id = load_static(level_dir, sid, scene, level)
                static_cache[(scene, level, sid)] = by_id
            object_ids = sorted(by_id)
            if not object_ids:
                continue
            for frame in range(1, FRAMES + 1):
                frame_dir = sample_dir / "dynamic" / str(frame)
                dyn_json_dir = frame_dir / "object_dynamicjson"
                seg_dir = frame_dir / "object_segment"
                for object_id in object_ids:
                    dyn_path = dyn_json_dir / f"{object_id}.json"
                    if not dyn_path.is_file():
                        missing_dynamic_json += 1
                        err(f"{scene}/L{level}/{sid}/dynamic/{frame}: 缺少object_dynamicjson/{object_id}.json")
                    seg_path = seg_dir / f"{object_id}.npz"
                    if not seg_path.is_file():
                        missing_segment += 1
                        err(f"{scene}/L{level}/{sid}/dynamic/{frame}: 缺少object_segment/{object_id}.npz")
                        continue
                    has_pixels = validate_segment_npz(
                        seg_path,
                        f"{scene}/L{level}/{sid}/dynamic/{frame}/object_segment/{object_id}.npz",
                        object_id,
                        frame,
                    )
                    segment_checked += 1
                    if not has_pixels:
                        segment_zero += 1

        stats[f"{scene}/L{level}_seg_zero"] = segment_zero
        if segment_zero:
            warn(f"{scene}/L{level}: 分割mask全零 {segment_zero} 个")
        emit(
            f"  {scene}/L{level}: depth={depth_checked}, segment={segment_checked}, "
            f"seg全零={segment_zero}, 缺dynamic_json={missing_dynamic_json}, 缺segment={missing_segment}"
        )


# ========== Phase 4: 跨视角一致性 ==========
emit("\n--- Phase 4: 跨视角一致性 ---")
for scene in SCENES:
    nv = VIEW_COUNT[scene]
    for level in sorted(PHYSICAL_LEVEL_TARGETS[scene]):
        sample_ids = level_samples.get((scene, level), [])
        if not sample_ids:
            continue
        level_dir = scene_levels[scene][level]
        events: list[list[int]] = []
        tail: list[int] = []
        for start in range(0, len(sample_ids), nv):
            group = sample_ids[start : start + nv]
            if len(group) == nv:
                events.append(group)
            else:
                tail = group
        if tail:
            err(f"{scene}/L{level}: 跨视角分组尾巴不足一组: {tail}, 视角数={nv}")

        issues_before = len(errors)
        checked_events = [events[i] for i in event_indices(len(events))]
        for ev in checked_events:
            views = [view_cache.get((scene, level, sid)) for sid in ev]
            valid_views = [v for v in views if v is not None]
            if len(valid_views) != nv:
                err(f"{scene}/L{level} 事件{ev}: 缺少可用view_name {views}")
            if len(set(valid_views)) != len(valid_views):
                err(f"{scene}/L{level} 事件{ev}: 视角重复 {views}")
            if set(valid_views) != EXPECTED_VIEWS[scene]:
                err(f"{scene}/L{level} 事件{ev}: 视角集合={sorted(valid_views)}, 期望 {sorted(EXPECTED_VIEWS[scene])}")

            statics = []
            for sid in ev:
                by_id = static_cache.get((scene, level, sid))
                if by_id is None:
                    by_id = load_static(level_dir, sid, scene, level)
                    static_cache[(scene, level, sid)] = by_id
                statics.append((sid, by_id))
            ref_sid, ref_static = statics[0]
            ref_ids = set(ref_static)
            for sid, current_static in statics[1:]:
                cur_ids = set(current_static)
                if cur_ids != ref_ids:
                    err(
                        f"{scene}/L{level} 事件{ev}: object_id集合不一致 "
                        f"{ref_sid}={sorted(ref_ids)} vs {sid}={sorted(cur_ids)}"
                    )
                    continue
                for object_id in sorted(ref_ids):
                    ref_obj = ref_static[object_id]
                    cur_obj = current_static[object_id]
                    for key in STATIC_COMPARE_KEYS:
                        if key not in ref_obj and key not in cur_obj:
                            continue
                        if key not in ref_obj or key not in cur_obj:
                            err(f"{scene}/L{level} 事件{ev}: object {object_id}.{key} 字段缺失")
                            continue
                        if not same_value(ref_obj[key], cur_obj[key]):
                            err(
                                f"{scene}/L{level} 事件{ev}: object {object_id}.{key}不一致 "
                                f"{ref_sid}={ref_obj[key]!r} vs {sid}={cur_obj[key]!r}"
                            )

            for frame in CHECK_FRAME_NUMS:
                dyn_by_sid: list[tuple[int, dict[int, dict[str, Any]]]] = []
                for sid in ev:
                    by_id = static_cache.get((scene, level, sid), {})
                    expected_ids = set(by_id)
                    dyn_dir = level_dir / str(sid) / "dynamic" / str(frame) / "object_dynamicjson"
                    current: dict[int, dict[str, Any]] = {}
                    if not dyn_dir.is_dir():
                        err(f"{scene}/L{level}/{sid}/dynamic/{frame}: 缺少object_dynamicjson/")
                        dyn_by_sid.append((sid, current))
                        continue
                    for object_id in sorted(expected_ids):
                        dyn_path = dyn_dir / f"{object_id}.json"
                        if not dyn_path.is_file():
                            err(f"{scene}/L{level}/{sid}/dynamic/{frame}: 缺少object_dynamicjson/{object_id}.json")
                            continue
                        data = read_json(dyn_path, f"{scene}/L{level}/{sid}/dynamic/{frame}/object_dynamicjson/{object_id}.json")
                        if isinstance(data, dict):
                            current[object_id] = data
                    dyn_by_sid.append((sid, current))

                ref_sid, ref_dyn = dyn_by_sid[0]
                ref_ids = set(ref_dyn)
                for sid, cur_dyn in dyn_by_sid[1:]:
                    cur_ids = set(cur_dyn)
                    if cur_ids != ref_ids:
                        err(
                            f"{scene}/L{level} 事件{ev} frame={frame}: dynamic object集合不一致 "
                            f"{ref_sid}={sorted(ref_ids)} vs {sid}={sorted(cur_ids)}"
                        )
                        continue
                    for object_id in sorted(ref_ids):
                        for key in DYNAMIC_COMPARE_KEYS:
                            if key not in ref_dyn[object_id] or key not in cur_dyn[object_id]:
                                err(f"{scene}/L{level} 事件{ev} frame={frame}: object {object_id}.{key}字段缺失")
                                continue
                            if not same_value(ref_dyn[object_id][key], cur_dyn[object_id][key], tol=1e-4):
                                err(
                                    f"{scene}/L{level} 事件{ev} frame={frame}: object {object_id}.{key}不一致 "
                                    f"{ref_sid}={ref_dyn[object_id][key]!r} vs {sid}={cur_dyn[object_id][key]!r}"
                                )

        issues = len(errors) - issues_before
        emit(f"  {scene}/L{level}: 事件={len(events)}, 抽检={len(checked_events)}, 跨视角问题={issues}")


# ========== 输出报告 ==========
emit("\n" + "=" * 70)
emit("验证报告摘要")
emit("=" * 70)
emit(f"总错误数: {len(errors)}")
emit(f"总警告数: {len(warnings)}")

if errors:
    emit(f"\n--- 错误 ({len(errors)}条) ---")
    for message in errors:
        emit(f"  ERROR {message}")

if warnings:
    emit(f"\n--- 警告 ({len(warnings)}条) ---")
    for message in warnings:
        emit(f"  WARN {message}")

emit("\n--- 统计 ---")
for key, value in sorted(stats.items()):
    emit(f"  {key}: {value}")

emit(f"\n报告已写入: {REPORT_PATH.resolve()}")
REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

if errors:
    sys.exit(1)
