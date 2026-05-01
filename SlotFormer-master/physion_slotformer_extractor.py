"""Physion evaluator adapter for SlotFormer/STEVE features.

Run from the SlotFormer repo root with:
physion_feature_extract --model_class physion_slotformer_extractor.SlotFormerSTEVEFeatureExtractor ...
"""

import os
import sys
from pathlib import Path

import torch
from torchvision import transforms

from physion_evaluator.feature_extract_interface import PhysionFeatureExtractor


REPO_ROOT = Path(__file__).resolve().parent
BASE_SLOTS_DIR = REPO_ROOT / "slotformer" / "base_slots"
if str(BASE_SLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_SLOTS_DIR))

from configs.steve_physion_params import SlotFormerParams  # noqa: E402
from models import build_model  # noqa: E402


class _VideoTransform:
    def __init__(self, resolution):
        self.frame_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __call__(self, images):
        return torch.stack([
            self.frame_transform(img.convert("RGB")) for img in images
        ], dim=0)


class SlotFormerSTEVEFeatureExtractor(PhysionFeatureExtractor):
    """Use SlotFormer's pretrained STEVE encoder as Physion features."""

    def __init__(self):
        super().__init__()
        params = SlotFormerParams()
        params.gpus = 1

        weight_path = os.environ.get(
            "SLOTFORMER_STEVE_WEIGHT",
            str(REPO_ROOT / "pretrained" / "steve_physion_params" / "model_10.pth"),
        )
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"STEVE weight not found: {weight_path}. "
                "Set SLOTFORMER_STEVE_WEIGHT to override it."
            )

        self.resolution = params.resolution
        self.model = build_model(params)
        checkpoint = torch.load(weight_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.testing = True
        self.model.eval()

    def transform(self):
        frame_gap = int(os.environ.get("PHYSION_FRAME_GAP", "150"))
        num_frames = int(os.environ.get("PHYSION_NUM_FRAMES", "4"))
        return _VideoTransform(self.resolution), frame_gap, num_frames

    @torch.no_grad()
    def extract_features(self, videos):
        device = next(self.model.parameters()).device
        videos = videos.to(device)
        out = self.model({"img": videos})
        slots = out["slots"]
        return slots.flatten(2)
