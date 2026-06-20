from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import Detect


def _rebuild_save_list(layers: nn.Sequential) -> list[int]:
    """Recompute the feature-cache indices used by Ultralytics' graph forward."""
    saved: set[int] = set()
    for module in layers:
        sources = [module.f] if isinstance(module.f, int) else list(module.f)
        for source in sources:
            if source != -1:
                saved.add(int(source))
    return sorted(saved)


def compact_p251_to_p3p4(model: nn.Module) -> nn.Module:
    """Remove only the bottom-up P5 detection path from a trained p251 model.

    Layers 0-19, including the complete deep backbone and top-down P3/P4 neck, are retained byte-for-byte. Layers 20-22
    and the third Detect tower are
    removed. The first two box/classification towers keep their trained weights.
    """
    model = model.float().cpu().eval()
    if not hasattr(model, "model") or len(model.model) < 24:
        raise ValueError(f"Expected a YOLO11 graph with at least 24 layers, got {len(getattr(model, 'model', []))}")

    old_head = model.model[-1]
    if not isinstance(old_head, Detect) or old_head.nl != 3:
        raise TypeError(
            f"Expected a three-level Detect head, got {type(old_head)!r} with nl={getattr(old_head, 'nl', None)}"
        )

    retained = list(model.model[:20])
    new_head = copy.deepcopy(old_head)
    new_head.nl = 2
    new_head.cv2 = nn.ModuleList(list(new_head.cv2[:2]))
    new_head.cv3 = nn.ModuleList(list(new_head.cv3[:2]))
    if new_head.end2end:
        new_head.one2one_cv2 = nn.ModuleList(list(new_head.one2one_cv2[:2]))
        new_head.one2one_cv3 = nn.ModuleList(list(new_head.one2one_cv3[:2]))

    new_head.i = 20
    new_head.f = [16, 19]
    new_head.stride = old_head.stride[:2].detach().clone()
    new_head.anchors = torch.empty(0)
    new_head.strides = torch.empty(0)
    new_head.shape = None
    new_head.dynamic = True
    new_head.np = sum(parameter.numel() for parameter in new_head.parameters())

    retained.append(new_head)
    model.model = nn.Sequential(*retained)
    model.save = _rebuild_save_list(model.model)
    model.stride = new_head.stride
    model.p3p4_compact = True
    model.compaction_metadata = {
        "removed_layers": [20, 21, 22],
        "detect_sources": [16, 19],
        "detect_strides": [float(value) for value in new_head.stride.cpu()],
    }

    # Force the first inference call to rebuild anchors for 80x80 and 40x40 only.
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 640, 640))
    raw = output[1] if isinstance(output, tuple) else output
    if not isinstance(raw, dict) or raw["scores"].shape[-1] != 8000:
        raise RuntimeError(f"Unexpected compact output: {type(raw)!r}, scores={getattr(raw, 'shape', None)}")

    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters >= 2_000_000:
        raise RuntimeError(f"P3/P4 compact model did not reach the sub-2M target: {parameters:,}")
    return model


def save_compact_checkpoint(model: nn.Module, source_checkpoint: str | Path, destination: str | Path) -> Path:
    """Save the changed graph through Ultralytics so it can be validated normally."""
    from ultralytics import YOLO

    source_checkpoint = Path(source_checkpoint)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapper = YOLO(str(source_checkpoint))
    wrapper.model = model
    if isinstance(getattr(wrapper, "ckpt", None), dict):
        # Ultralytics validation prefers the checkpoint EMA when present. Replace
        # it as well, otherwise reloading the compact checkpoint silently restores
        # the unpruned source graph.
        compact_ema = copy.deepcopy(model).half()
        for parameter in compact_ema.parameters():
            parameter.requires_grad_(False)
        wrapper.ckpt["ema"] = compact_ema
        wrapper.ckpt["updates"] = 0
    wrapper.save(destination)
    return destination
