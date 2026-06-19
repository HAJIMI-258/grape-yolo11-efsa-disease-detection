from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/highsource7/finegrained_detect.py"
MODEL_CFG = ROOT / "scripts/highsource7/yolo11n_width20_highsource7.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("finegrained_detect", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_finegrained_head_only_widens_classification_towers():
    module = _load_module()
    model = YOLO(MODEL_CFG).model
    old_head = model.model[-1]
    old_box_ids = [id(branch) for branch in old_head.cv2]
    old_params = sum(parameter.numel() for parameter in model.parameters())

    module.replace_detect_with_finegrained(model, cls_channels=96)
    head = model.model[-1]
    new_params = sum(parameter.numel() for parameter in model.parameters())

    assert head.cls_channels == 96
    assert [id(branch) for branch in head.cv2] == old_box_ids
    assert new_params > old_params
    assert 1_200_000 <= new_params <= 1_350_000

    model.train()
    outputs = model(torch.randn(1, 3, 640, 640))
    assert outputs["scores"].shape[1] == 7
    assert outputs["boxes"].shape[1] == 64
