from __future__ import annotations

import math
from typing import Any

import numpy as np


CONTACT_EPS = 1e-5


def _norm(vec: Any) -> float:
    if vec is None:
        return 0.0
    return float(np.linalg.norm(np.array(vec, dtype=np.float64)))


def _contact_radius(spec: Any) -> float:
    if getattr(spec, "radius", None) is not None:
        return float(spec.radius)
    if getattr(spec, "size", None) is not None:
        sx, sy, _ = spec.size
        return 0.5 * float(math.hypot(sx, sy))
    return 0.0


def _letter_map(dynamic_ids: list[int]) -> dict[int, str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    return {object_id: letters[idx] for idx, object_id in enumerate(dynamic_ids)}


def _speed(state: dict[str, Any]) -> float:
    return _norm(state.get("velocity"))


def _angular_speed(state: dict[str, Any]) -> float:
    return _norm(state.get("angular_velocity"))


def _object_properties(spec: Any) -> dict[str, Any]:
    return {
        "object_type": spec.object_type,
        "static": spec.static,
        "radius": spec.radius,
        "size": None if spec.size is None else list(spec.size),
        "height": spec.height,
        "mass": spec.mass,
        "lateralFriction": spec.lateral_friction,
        "rollingFriction": spec.rolling_friction,
        "spinningFriction": spec.spinning_friction,
        "restitution": spec.restitution,
    }


def _active_pairs(force_matrix: list[list[Any]], object_ids: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left_idx, object_a in enumerate(object_ids):
        for object_b in object_ids[left_idx + 1:]:
            force_ab = _norm(force_matrix[object_a][object_b]) if object_a < len(force_matrix) and object_b < len(force_matrix[object_a]) else 0.0
            force_ba = _norm(force_matrix[object_b][object_a]) if object_b < len(force_matrix) and object_a < len(force_matrix[object_b]) else 0.0
            if force_ab > CONTACT_EPS or force_ba > CONTACT_EPS:
                pairs.append((object_a, object_b))
    return pairs


def compute_physics_labels(sample: Any, frame_states: list[dict[int, dict[str, Any]]], force_payloads: list[dict[str, Any]], fps: int) -> dict[str, Any]:
    specs_by_id = {spec.object_id: spec for spec in sample.objects}
    object_ids = sorted(specs_by_id)
    dynamic_ids = [object_id for object_id in object_ids if not specs_by_id[object_id].static]
    static_ids = [object_id for object_id in object_ids if specs_by_id[object_id].static]
    letters = _letter_map(dynamic_ids)

    labels: dict[str, Any] = {
        "level_id": sample.level_id,
        "level_name": sample.level_name,
        "subtask": sample.subtask,
        "main_variable": sample.main_variable,
        "object_count": len(dynamic_ids),
        "dynamic_object_types": [specs_by_id[object_id].object_type for object_id in dynamic_ids],
        "initial_positions": {str(object_id): specs_by_id[object_id].position for object_id in dynamic_ids},
        "initial_velocities": {str(object_id): specs_by_id[object_id].velocity for object_id in dynamic_ids},
        "initial_object_properties": {str(object_id): _object_properties(specs_by_id[object_id]) for object_id in object_ids},
    }
    metadata = getattr(sample, "metadata", None)
    if metadata:
        labels.update(metadata)
        labels["sample_metadata"] = metadata

    contact_pair_sequence: list[dict[str, Any]] = []
    seen_steps: set[tuple[int, int, int]] = set()
    first_dynamic_contacts: dict[tuple[int, int], int] = {}
    dynamic_dynamic_contact_count = 0
    dynamic_static_contact_count = 0

    for frame_idx, payload in enumerate(force_payloads):
        force_matrix = payload["force_matrix"]
        for object_a, object_b in _active_pairs(force_matrix, object_ids):
            spec_a = specs_by_id[object_a]
            spec_b = specs_by_id[object_b]
            if spec_a.static and spec_b.static:
                continue
            pair_key = (min(object_a, object_b), max(object_a, object_b))
            event_key = (frame_idx, pair_key[0], pair_key[1])
            if event_key not in seen_steps:
                seen_steps.add(event_key)
                contact_pair_sequence.append({
                    "frame": frame_idx + 1,
                    "time": frame_idx / fps,
                    "pair": [pair_key[0], pair_key[1]],
                    "pair_names": [specs_by_id[pair_key[0]].name, specs_by_id[pair_key[1]].name],
                })
            if spec_a.static or spec_b.static:
                dynamic_static_contact_count += 1
            else:
                dynamic_dynamic_contact_count += 1
                first_dynamic_contacts.setdefault(pair_key, frame_idx)

    labels["contact_pair_sequence"] = contact_pair_sequence
    labels["wall_contact_sequence"] = [
        event for event in contact_pair_sequence
        if any(specs_by_id[object_id].object_type in {"wall", "ground"} for object_id in event["pair"])
    ]
    labels["total_dynamic_dynamic_contact_count"] = dynamic_dynamic_contact_count
    labels["total_dynamic_static_contact_count"] = dynamic_static_contact_count
    labels["no_dynamic_collision"] = dynamic_dynamic_contact_count == 0

    for (object_a, object_b), frame_idx in first_dynamic_contacts.items():
        label_a = letters.get(object_a, str(object_a))
        label_b = letters.get(object_b, str(object_b))
        labels[f"first_contact_frame_{label_a}{label_b}"] = frame_idx + 1
        labels[f"first_contact_time_{label_a}{label_b}"] = frame_idx / fps

    min_distance = float("inf")
    min_frame = 0
    closest_pair: tuple[int, int] | None = None
    rel_velocity_at_closest: list[float] | None = None
    for frame_idx, states in enumerate(frame_states):
        for idx, object_a in enumerate(dynamic_ids):
            for object_b in dynamic_ids[idx + 1:]:
                pos_a = np.array(states[object_a]["position"], dtype=np.float64)
                pos_b = np.array(states[object_b]["position"], dtype=np.float64)
                clearance = float(np.linalg.norm(pos_a - pos_b) - _contact_radius(specs_by_id[object_a]) - _contact_radius(specs_by_id[object_b]))
                if clearance < min_distance:
                    min_distance = clearance
                    min_frame = frame_idx
                    closest_pair = (object_a, object_b)
                    vel_a = np.array(states[object_a]["velocity"], dtype=np.float64)
                    vel_b = np.array(states[object_b]["velocity"], dtype=np.float64)
                    rel_velocity_at_closest = (vel_a - vel_b).astype(float).tolist()

    if closest_pair is not None:
        labels["min_pairwise_distance"] = min_distance
        labels["min_pairwise_distance_frame"] = min_frame + 1
        labels["closest_approach_time"] = min_frame / fps
        labels["closest_approach_frame"] = min_frame + 1
        labels["closest_pair"] = list(closest_pair)
        labels["relative_velocity_at_closest_approach"] = rel_velocity_at_closest

    if frame_states:
        final_states = frame_states[-1]
        for object_id in dynamic_ids:
            label = letters[object_id]
            velocity = final_states[object_id]["velocity"]
            speed = _norm(velocity)
            labels[f"final_position_{label}"] = final_states[object_id]["position"]
            labels[f"final_velocity_{label}"] = velocity
            labels[f"final_linear_speed_{label}"] = speed
            labels[f"is_static_at_end_{label}"] = speed < 0.05

    if len(dynamic_ids) == 1 and frame_states:
        object_id = dynamic_ids[0]
        spec = specs_by_id[object_id]
        initial_state = frame_states[0][object_id]
        final_state = frame_states[-1][object_id]
        initial_position = np.array(initial_state["position"], dtype=np.float64)
        final_position = np.array(final_state["position"], dtype=np.float64)
        final_speed = _speed(final_state)
        final_angular_speed = _angular_speed(final_state)

        labels["final_position"] = final_state["position"]
        labels["final_velocity"] = final_state["velocity"]
        labels["final_linear_speed"] = final_speed
        labels["final_angular_speed"] = final_angular_speed
        labels["is_static_at_end"] = final_speed < 0.05 and final_angular_speed < 0.1
        labels["displacement_vector"] = (final_position - initial_position).astype(float).tolist()
        labels["travel_distance"] = float(np.linalg.norm((final_position - initial_position)[:2]))
        labels["max_height_after_release"] = max(float(states[object_id]["position"][2]) for states in frame_states)
        labels["object_lateral_friction"] = float(spec.lateral_friction)
        labels["object_rolling_friction"] = float(spec.rolling_friction)
        labels["object_spinning_friction"] = float(spec.spinning_friction)
        labels["effective_lateral_friction"] = float(spec.lateral_friction)

        first_static_frame: int | None = None
        first_static_id: int | None = None
        for event in contact_pair_sequence:
            pair = event["pair"]
            if object_id in pair:
                other_id = pair[0] if pair[1] == object_id else pair[1]
                if other_id in static_ids:
                    first_static_frame = int(event["frame"]) - 1
                    first_static_id = other_id
                    break
        if first_static_frame is not None:
            first_contact_state = frame_states[first_static_frame][object_id]
            labels["first_contact_frame"] = first_static_frame + 1
            labels["first_contact_time"] = first_static_frame / fps
            labels["landing_position"] = first_contact_state["position"]
            labels["rebound_height"] = max(
                float(states[object_id]["position"][2])
                for states in frame_states[first_static_frame:]
            )
            if first_static_id is not None:
                static_spec = specs_by_id[first_static_id]
                labels["contact_surface_object_id"] = first_static_id
                labels["contact_surface_type"] = static_spec.object_type
                labels["contact_surface_lateral_friction"] = float(static_spec.lateral_friction)
                labels["contact_surface_rolling_friction"] = float(static_spec.rolling_friction)
                labels["contact_surface_spinning_friction"] = float(static_spec.spinning_friction)
                labels["contact_surface_restitution"] = float(static_spec.restitution)
                labels["effective_lateral_friction"] = float(spec.lateral_friction * static_spec.lateral_friction)
                labels["effective_restitution"] = float(spec.restitution * static_spec.restitution)

        stop_frame: int | None = None
        for frame_idx, states in enumerate(frame_states):
            if _speed(states[object_id]) < 0.05 and _angular_speed(states[object_id]) < 0.1:
                stop_frame = frame_idx
                break
        if stop_frame is None:
            stop_frame = len(frame_states) - 1
        labels["stop_frame"] = stop_frame + 1
        labels["stop_time"] = stop_frame / fps

        contact_radius = _contact_radius(spec)
        rolling_frame: int | None = None
        if contact_radius > 0:
            for frame_idx, states in enumerate(frame_states):
                linear = _speed(states[object_id])
                angular = _angular_speed(states[object_id])
                if linear > 0.05 and abs(linear - angular * contact_radius) / max(linear, 1e-6) < 0.15:
                    rolling_frame = frame_idx
                    break
        labels["rolling_transition_frame"] = None if rolling_frame is None else rolling_frame + 1
        labels["rolling_transition_time"] = None if rolling_frame is None else rolling_frame / fps

    # Keep common S5/S6 two-body aliases and S7 chain aliases easy to consume.
    if len(dynamic_ids) >= 2:
        first_a, first_b = dynamic_ids[0], dynamic_ids[1]
        va = np.array(specs_by_id[first_a].velocity, dtype=np.float64)
        vb = np.array(specs_by_id[first_b].velocity, dtype=np.float64)
        labels["relative_velocity_ab_before"] = (va - vb).astype(float).tolist()
    if len(dynamic_ids) >= 3:
        second_b, second_c = dynamic_ids[1], dynamic_ids[2]
        vb = np.array(specs_by_id[second_b].velocity, dtype=np.float64)
        vc = np.array(specs_by_id[second_c].velocity, dtype=np.float64)
        labels["relative_velocity_bc_before"] = (vb - vc).astype(float).tolist()

    labels["static_object_ids"] = static_ids
    return labels
