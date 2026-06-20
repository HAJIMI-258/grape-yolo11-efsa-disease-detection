from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/highsource7/p3p4_compaction.py"
MODEL_CFG = ROOT / "ultralytics/cfg/models/11/yolo11.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("p3p4_compaction", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_p3p4_compaction_removes_p5_path_and_preserves_first_two_towers():
    module = _load_module()
    model = YOLO(MODEL_CFG).model.float().eval()
    old_head = model.model[-1]
    old_box = [branch.state_dict() for branch in old_head.cv2[:2]]
    old_cls = [branch.state_dict() for branch in old_head.cv3[:2]]
    before = sum(parameter.numel() for parameter in model.parameters())

    compact = module.compact_p251_to_p3p4(model)
    after = sum(parameter.numel() for parameter in compact.parameters())
    head = compact.model[-1]

    assert len(compact.model) == 21
    assert head.nl == 2
    assert head.f == [16, 19]
    assert compact.save == [4, 6, 13, 16, 19]
    assert after < before
    assert after < 2_000_000

    for branch, expected in zip(head.cv2, old_box):
        for key, value in branch.state_dict().items():
            assert torch.equal(value, expected[key])
    for branch, expected in zip(head.cv3, old_cls):
        for key, value in branch.state_dict().items():
            assert torch.equal(value, expected[key])

    compact.eval()
    with torch.no_grad():
        decoded, raw = compact(torch.zeros(1, 3, 640, 640))
    assert raw["scores"].shape[-1] == 8000
    assert raw["boxes"].shape[-1] == 8000
    assert decoded.shape[-1] == 8000
