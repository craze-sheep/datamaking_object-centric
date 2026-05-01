#!/usr/bin/env python3
"""Create a tiny evaluator-style evidence bundle from local Kubric videos.

Outputs:
  - features.hdf5
  - readout_results.json
  - evaluator.log

This is a smoke test for the adapter / feature extraction path, not the full
24.7GB official Physion benchmark.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import cv2
from PIL import Image

from physion_slotformer_extractor import SlotFormerSTEVEFeatureExtractor


def load_frames(video_path: Path) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb.astype(np.uint8)))
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return frames


def encode_label(event_type: str) -> int:
    mapping = {
        "bounce": 0,
        "slide": 1,
        "collision": 2,
    }
    return mapping[event_type]


def main() -> None:
    data_root = Path(
        os.environ.get(
            "PHYSION_SMOKETEST_DATA",
            "/root/project/kubric-main/output/mini_physics_clean",
        )
    )
    output_dir = Path(
        os.environ.get(
            "PHYSION_SMOKETEST_DIR",
            "/root/project/SlotFormer-master/artifacts/physion_evaluator_smoketest",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = SlotFormerSTEVEFeatureExtractor()
    transform, frame_gap, num_frames = extractor.transform()
    scenario_dirs = sorted(
        [p for p in data_root.iterdir() if p.is_dir()]
    )

    feature_list = []
    pooled_list = []
    label_list = []
    event_names = []
    scenario_names = []

    for scenario_dir in scenario_dirs:
        label_path = scenario_dir / "labels.json"
        rgb_path = scenario_dir / "rgb.mp4"
        with open(label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        event_type = labels["event_labels"]["event_type"]

        pil_frames = load_frames(rgb_path)
        frame_indices = np.linspace(0, len(pil_frames) - 1, num_frames, dtype=int)
        selected = [pil_frames[i] for i in frame_indices]
        video_tensor = transform(selected).unsqueeze(0)

        with torch.no_grad():
            features = extractor.extract_features(video_tensor).cpu().numpy()[0]

        feature_list.append(features)
        pooled_list.append(features.mean(axis=0))
        label_list.append(encode_label(event_type))
        event_names.append(event_type)
        scenario_names.append(labels["scenario"])

    features = np.stack(feature_list, axis=0)
    pooled = np.stack(pooled_list, axis=0)
    labels = np.array(label_list, dtype=np.int64)

    # Tiny nearest-centroid readout over the same smoke-test set.
    class_centroids = {}
    predictions = []
    for label in np.unique(labels):
        class_centroids[int(label)] = pooled[labels == label].mean(axis=0)
    for vec in pooled:
        best_label = min(
            class_centroids,
            key=lambda cls: float(np.linalg.norm(vec - class_centroids[cls])),
        )
        predictions.append(int(best_label))
    predictions = np.array(predictions, dtype=np.int64)
    accuracy = float((predictions == labels).mean())

    h5_path = output_dir / "features.hdf5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("pooled_features", data=pooled)
        f.create_dataset("labels", data=labels)
        f.create_dataset("predictions", data=predictions)
        f.create_dataset("event_names", data=np.array(event_names, dtype="S32"))
        f.create_dataset("scenario_names", data=np.array(scenario_names, dtype="S64"))

    result = {
        "note": "Mini smoke test over local Kubric videos using the SlotFormer Physion adapter.",
        "data_root": str(data_root),
        "frame_gap": int(frame_gap),
        "num_frames": int(num_frames),
        "features_shape": list(features.shape),
        "pooled_features_shape": list(pooled.shape),
        "labels": labels.tolist(),
        "predictions": predictions.tolist(),
        "event_names": event_names,
        "scenario_names": scenario_names,
        "readout": {
            "type": "nearest_centroid",
            "accuracy_on_smoketest_set": accuracy,
        },
        "feature_file": str(h5_path),
    }
    with open(output_dir / "readout_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with open(output_dir / "evaluator.log", "w", encoding="utf-8") as f:
        f.write("Physion evaluator smoke test\n")
        f.write(f"data_root: {data_root}\n")
        f.write(f"frame_gap: {frame_gap}\n")
        f.write(f"num_frames: {num_frames}\n")
        f.write(f"features_shape: {tuple(features.shape)}\n")
        f.write(f"pooled_features_shape: {tuple(pooled.shape)}\n")
        f.write(f"event_names: {event_names}\n")
        f.write(f"scenario_names: {scenario_names}\n")
        f.write(f"predictions: {predictions.tolist()}\n")
        f.write(f"accuracy_on_smoketest_set: {accuracy:.4f}\n")
        f.write(f"saved_features: {h5_path}\n")

    print(f"Saved Physion evaluator smoke test to {output_dir}")
    print(f"features_shape={tuple(features.shape)}")
    print(f"accuracy_on_smoketest_set={accuracy:.4f}")


if __name__ == "__main__":
    main()
