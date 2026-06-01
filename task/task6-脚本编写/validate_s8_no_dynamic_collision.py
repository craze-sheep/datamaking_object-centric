#!/usr/bin/env python3
"""Validate S8 negative samples with direct PyBullet collision checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_s8_dataset as s8  # noqa: E402


AFFECTED_LEVELS = [2, 3, 5, 7, 10, 11, 12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=AFFECTED_LEVELS)
    parser.add_argument("--samples_per_level", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    return parser.parse_args()


def wxyz_to_xyzw(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (quaternion[1], quaternion[2], quaternion[3], quaternion[0])


def add_pybullet_body(client_id: int, spec: s8.ObjectSpec) -> int:
    mass = 0.0 if spec.static else float(spec.mass or 1.0)
    if spec.object_type == "sphere":
        collision_id = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=spec.radius,
            physicsClientId=client_id,
        )
    elif spec.object_type in {"cube", "ground", "wall"}:
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[value / 2.0 for value in spec.size],
            physicsClientId=client_id,
        )
    elif spec.object_type == "cylinder":
        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=spec.radius,
            height=spec.height,
            physicsClientId=client_id,
        )
    else:
        raise ValueError(f"Unsupported object type: {spec.object_type}")

    body_id = p.createMultiBody(
        mass,
        collision_id,
        -1,
        spec.position,
        wxyz_to_xyzw(spec.quaternion),
        useMaximalCoordinates=True,
        physicsClientId=client_id,
    )
    p.changeDynamics(
        body_id,
        -1,
        lateralFriction=spec.lateral_friction,
        rollingFriction=spec.rolling_friction,
        spinningFriction=spec.spinning_friction,
        restitution=spec.restitution,
        contactProcessingThreshold=0,
        physicsClientId=client_id,
    )
    p.resetBaseVelocity(
        body_id,
        linearVelocity=spec.velocity,
        angularVelocity=spec.angular_velocity,
        physicsClientId=client_id,
    )
    return body_id


def apply_impulse_if_needed(
    client_id: int,
    sample: s8.SampleSpec,
    bodies_by_object_id: dict[int, int],
    step: int,
) -> None:
    metadata = sample.metadata or {}
    if not metadata.get("external_impulse_enabled"):
        return

    steps_per_frame = s8.SIM_HZ // s8.FPS
    impulse_start_step = int(float(metadata.get("intervention_time", -1.0)) * s8.SIM_HZ)
    impulse_duration_steps = max(1, int(metadata.get("duration_frames", 1)) * steps_per_frame)
    if not impulse_start_step <= step < impulse_start_step + impulse_duration_steps:
        return

    target_object = int(metadata.get("target_object", -1))
    if target_object not in bodies_by_object_id:
        return

    target_spec = next(spec for spec in sample.objects if spec.object_id == target_object)
    target_body = bodies_by_object_id[target_object]
    linear_velocity, angular_velocity = p.getBaseVelocity(target_body, physicsClientId=client_id)
    impulse = np.array(metadata.get("impulse_vector", (0.0, 0.0, 0.0)), dtype=np.float64)
    delta_v = impulse / max(float(target_spec.mass or 1.0), 1e-6) / impulse_duration_steps
    p.resetBaseVelocity(
        target_body,
        linearVelocity=(np.array(linear_velocity, dtype=np.float64) + delta_v).astype(float).tolist(),
        angularVelocity=angular_velocity,
        physicsClientId=client_id,
    )


def count_dynamic_dynamic_contacts(sample: s8.SampleSpec) -> tuple[int, set[tuple[int, int]]]:
    client_id = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=client_id)
        p.setGravity(*s8.GRAVITY, physicsClientId=client_id)
        p.setTimeStep(1.0 / s8.SIM_HZ, physicsClientId=client_id)

        bodies_by_object_id: dict[int, int] = {}
        object_id_by_body: dict[int, int] = {}
        dynamic_object_ids = set()
        for spec in sample.objects:
            body_id = add_pybullet_body(client_id, spec)
            bodies_by_object_id[spec.object_id] = body_id
            object_id_by_body[body_id] = spec.object_id
            if not spec.static:
                dynamic_object_ids.add(spec.object_id)

        contact_count = 0
        contact_pairs: set[tuple[int, int]] = set()
        total_steps = s8.NUM_FRAMES * (s8.SIM_HZ // s8.FPS)
        for step in range(total_steps):
            apply_impulse_if_needed(client_id, sample, bodies_by_object_id, step)
            for contact in p.getContactPoints(physicsClientId=client_id):
                object_a = object_id_by_body.get(int(contact[1]))
                object_b = object_id_by_body.get(int(contact[2]))
                normal_force = float(contact[9])
                if (
                    object_a in dynamic_object_ids
                    and object_b in dynamic_object_ids
                    and normal_force > 1e-6
                ):
                    contact_count += 1
                    contact_pairs.add(tuple(sorted((object_a, object_b))))
            p.stepSimulation(physicsClientId=client_id)
        return contact_count, contact_pairs
    finally:
        p.disconnect(client_id)


def validate_level(level_id: int, samples_per_level: int, seed: int) -> dict[str, Any]:
    samples = s8.take_samples(level_id, 1, seed, samples_per_level)[:samples_per_level]
    collided_samples = 0
    total_contacts = 0
    collided_pairs: set[tuple[int, int]] = set()
    for sample in samples:
        contact_count, contact_pairs = count_dynamic_dynamic_contacts(sample)
        total_contacts += contact_count
        if contact_count:
            collided_samples += 1
            collided_pairs.update(contact_pairs)

    sample_count = len(samples)
    return {
        "level_id": level_id,
        "sample_count": sample_count,
        "collided_samples": collided_samples,
        "total_contacts": total_contacts,
        "collision_rate": collided_samples / sample_count if sample_count else 0.0,
        "collided_pairs": sorted(collided_pairs),
    }


def main() -> None:
    args = parse_args()
    failed = False
    seeds = args.seeds if args.seeds is not None else [args.seed]
    for seed in seeds:
        for level_id in args.levels:
            result = validate_level(level_id, args.samples_per_level, seed)
            print(
                f"seed={seed} S8/L{level_id}: samples={result['sample_count']} "
                f"collided_samples={result['collided_samples']} "
                f"total_dynamic_dynamic_contacts={result['total_contacts']} "
                f"collision_rate={result['collision_rate']:.3f}"
            )
            if result["collided_samples"]:
                failed = True
                print(f"  collided_pairs={result['collided_pairs']}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
