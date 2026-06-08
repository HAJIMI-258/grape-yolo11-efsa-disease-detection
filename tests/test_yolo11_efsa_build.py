# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Smoke tests for the Grape-EFSA-YOLO11 method modules."""

from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import BiFPNFuse, CARAFEUp, CoordECA, EFSAEnhance


ROOT = Path(__file__).resolve().parents[1]


def test_efsa_modules_keep_expected_shapes():
    """Check the custom modules before building a full YOLO model."""
    x = torch.randn(1, 32, 24, 24)

    assert EFSAEnhance(32)(x).shape == x.shape
    assert CARAFEUp(32)(x).shape == (1, 32, 48, 48)
    assert CoordECA(32)(x).shape == x.shape

    p3 = torch.randn(1, 32, 48, 48)
    p4 = torch.randn(1, 64, 24, 24)
    fused = BiFPNFuse([32, 64], 40)([p3, p4])
    assert fused.shape == (1, 40, 48, 48)


def test_yolo11_efsa_yaml_builds():
    """Build the YOLO11 model from YAML to verify Ultralytics parser integration."""
    model = YOLO(ROOT / "configs/models/yolo11n_efsa_grape.yaml")
    assert model.model is not None
