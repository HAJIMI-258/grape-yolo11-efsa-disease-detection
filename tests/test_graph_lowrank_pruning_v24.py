from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from ultralytics.nn.modules.conv import Conv

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/highsource7"
MODULE_PATH = SCRIPT_DIR / "graph_lowrank_pruning_v24.py"
LOWRANK_PATH = SCRIPT_DIR / "lowrank_compression_v23.py"


def _load_modules():
    sys.path.insert(0, str(SCRIPT_DIR))
    lowrank_spec = importlib.util.spec_from_file_location("lowrank_compression_v23", LOWRANK_PATH)
    lowrank = importlib.util.module_from_spec(lowrank_spec)
    assert lowrank_spec.loader is not None
    sys.modules[lowrank_spec.name] = lowrank
    lowrank_spec.loader.exec_module(lowrank)

    spec = importlib.util.spec_from_file_location("graph_lowrank_pruning_v24", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return lowrank, module


def test_rank_importance_matches_lowrank_width():
    lowrank, module = _load_modules()
    block = Conv(32, 48, 3)
    factorized = lowrank._factorized_module(block, rank=12)

    score = module._rank_importance(factorized)

    assert score.shape == (12,)
    assert torch.isfinite(score).all()
    assert torch.all(score > 0)


def test_v24_sensitivity_protects_p3_p4_class_towers():
    _lowrank, module = _load_modules()

    assert module._candidate_sensitivity(23, "cv3.0.0.1") == float("inf")
    assert module._candidate_sensitivity(23, "cv3.1.1.1") == float("inf")
    assert module._candidate_sensitivity(23, "cv3.2.0.1") < float("inf")
    assert module._candidate_sensitivity(4, "conv") == float("inf")
