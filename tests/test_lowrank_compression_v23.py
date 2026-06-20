from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ultralytics.nn.modules.conv import Conv

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/highsource7/lowrank_compression_v23.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("lowrank_compression_v23", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lowrank_conv_preserves_geometry_and_reduces_parameters():
    module = _load_module()
    block = Conv(64, 64, 3, 2)
    block.eval()
    replacement = module._factorized_module(block, rank=32)
    replacement.eval()

    x = torch.randn(2, 64, 32, 32)
    with torch.no_grad():
        y = replacement(x)

    assert y.shape == (2, 64, 16, 16)
    assert replacement.spatial.stride == block.conv.stride
    assert replacement.spatial.padding == block.conv.padding
    assert replacement.spatial.dilation == block.conv.dilation
    original = block.conv.weight.numel()
    factorized = replacement.spatial.weight.numel() + replacement.pointwise.weight.numel()
    assert factorized < original


def test_progressive_factorization_can_lower_existing_rank():
    module = _load_module()
    block = Conv(64, 64, 3)
    first = module._factorized_module(block, rank=40)
    second = module._factorized_module(first, rank=24)

    assert isinstance(second, module.LowRankConv)
    assert second.rank == 24
    assert second.spatial.in_channels == 64
    assert second.pointwise.out_channels == 64
    assert second.spatial.weight.numel() + second.pointwise.weight.numel() < (
        first.spatial.weight.numel() + first.pointwise.weight.numel()
    )
