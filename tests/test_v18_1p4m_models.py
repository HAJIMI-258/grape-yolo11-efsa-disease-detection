from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn.modules import ESSE, GSConvns

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "configs/models/yolo11n_1p4m_realloc_highsource7_region.yaml",
    ROOT / "configs/models/yolo11n_1p4m_selective_gew_highsource7_region.yaml",
)


def test_v18_models_build_within_1p4m_budget():
    for config in CONFIGS:
        model = YOLO(config).model
        parameters = sum(parameter.numel() for parameter in model.parameters())
        assert 1_350_000 <= parameters <= 1_450_000, (config.name, parameters)


def test_selective_gew_is_restricted_to_one_gsconv_and_two_esse_blocks():
    model = YOLO(CONFIGS[1]).model
    assert sum(isinstance(module, GSConvns) for module in model.modules()) == 1
    assert sum(isinstance(module, ESSE) for module in model.modules()) == 2
