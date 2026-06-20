from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True)
class P5InternalProfile:
    """Internal-channel targets that preserve all three Detect scales."""

    name: str
    c3k_hidden: int
    p5_reg_hidden: int
    p5_cls_hidden: int
    min_reduction_percent: float


PROFILES = {
    # Keep the first pass conservative. V21 showed that deleting a scale is not
    # viable, so V22 only removes low-energy internal channels.
    "h56": P5InternalProfile("h56", c3k_hidden=56, p5_reg_hidden=56, p5_cls_hidden=56, min_reduction_percent=2.5),
    "h48": P5InternalProfile("h48", c3k_hidden=48, p5_reg_hidden=48, p5_cls_hidden=48, min_reduction_percent=5.5),
    "h40": P5InternalProfile("h40", c3k_hidden=40, p5_reg_hidden=48, p5_cls_hidden=48, min_reduction_percent=7.0),
}


@dataclass(frozen=True)
class InternalTarget:
    layer_index: int
    module_path: str
    target_channels: int


def _nested_module(module: nn.Module, path: str) -> nn.Module:
    current = module
    for token in path.split("."):
        if token.isdigit() and isinstance(current, (nn.ModuleList, nn.Sequential)):
            current = current[int(token)]
        else:
            current = getattr(current, token)
    return current


def _target_list(profile: P5InternalProfile) -> list[InternalTarget]:
    # Layer 8 and 22 are the deep C3k2/C3k blocks. Pruning only the two C3k
    # branch outputs preserves the surrounding C3k2 input/output channel counts.
    # The Detect head keeps P3/P4 intact and prunes only the P5 tower internals.
    return [
        InternalTarget(8, "m.0.cv1.conv", profile.c3k_hidden),
        InternalTarget(8, "m.0.cv2.conv", profile.c3k_hidden),
        InternalTarget(22, "m.0.cv1.conv", profile.c3k_hidden),
        InternalTarget(22, "m.0.cv2.conv", profile.c3k_hidden),
        InternalTarget(23, "cv2.2.0.conv", profile.p5_reg_hidden),
        InternalTarget(23, "cv2.2.1.conv", profile.p5_reg_hidden),
        InternalTarget(23, "cv3.2.0.1.conv", profile.p5_cls_hidden),
        InternalTarget(23, "cv3.2.1.1.conv", profile.p5_cls_hidden),
    ]


def _bn_scale_for_target(block: nn.Module, module_path: str, channels: int, device: torch.device) -> torch.Tensor:
    parent_path = module_path.rsplit(".", 1)[0] if "." in module_path else ""
    parent = _nested_module(block, parent_path) if parent_path else block
    bn = getattr(parent, "bn", None)
    if isinstance(bn, nn.BatchNorm2d) and bn.weight.numel() == channels:
        return bn.weight.detach().abs().to(device)
    return torch.ones(channels, device=device)


def _least_important_output_channels(block: nn.Module, module_path: str, prune_count: int) -> list[int]:
    conv = _nested_module(block, module_path)
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"Expected Conv2d at {module_path}, got {type(conv)!r}")
    if prune_count <= 0:
        return []
    if prune_count >= conv.out_channels:
        raise ValueError(f"Refusing to remove all {conv.out_channels} channels from {module_path}")

    weight_score = conv.weight.detach().float().flatten(1).pow(2).sum(1).sqrt()
    bn_scale = _bn_scale_for_target(block, module_path, conv.out_channels, weight_score.device)
    return (weight_score * bn_scale).argsort()[:prune_count].cpu().tolist()


def _refresh_metadata(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "np"):
            module.np = sum(parameter.numel() for parameter in module.parameters())
    head = model.model[-1]
    if hasattr(head, "shape"):
        head.shape = None
    if hasattr(head, "dynamic"):
        head.dynamic = True


def prune_yolo11n_p5_internal(
    model: nn.Module,
    profile: P5InternalProfile,
    *,
    example_inputs: torch.Tensor | None = None,
) -> nn.Module:
    """Prune P5 internals while preserving the three-scale Detect interface."""
    try:
        import torch_pruning as tp
    except ImportError as exc:
        raise RuntimeError("Install the pinned dependency with: pip install -r requirements-pruning.txt") from exc

    model = model.float().cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if example_inputs is None:
        example_inputs = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    example_inputs = example_inputs.to(next(model.parameters()).device)
    before = sum(parameter.numel() for parameter in model.parameters())

    for target in _target_list(profile):
        block = model.model[target.layer_index]
        root = _nested_module(block, target.module_path)
        if not isinstance(root, nn.Conv2d):
            raise TypeError(f"Layer {target.layer_index}.{target.module_path} must be Conv2d, got {type(root)!r}")
        if root.out_channels <= target.target_channels:
            continue

        prune_count = root.out_channels - target.target_channels
        with torch.enable_grad():
            graph = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
        indices = _least_important_output_channels(block, target.module_path, prune_count)
        group = graph.get_pruning_group(root, tp.prune_conv_out_channels, idxs=indices)
        if not graph.check_pruning_group(group):
            raise RuntimeError(f"Invalid pruning group at layer {target.layer_index}.{target.module_path}: {group}")
        group.prune()

    _refresh_metadata(model)
    with torch.no_grad():
        output = model(example_inputs)
    raw = output[1] if isinstance(output, tuple) else output
    if not isinstance(raw, dict) or raw["scores"].shape[-1] != 8400:
        raise RuntimeError(f"V22 must keep all three scales and 8400 anchors, got {type(raw)!r}")

    after = sum(parameter.numel() for parameter in model.parameters())
    reduction = 100.0 * (before - after) / before
    if reduction < profile.min_reduction_percent:
        raise RuntimeError(
            f"V22 {profile.name} reduction too small: {reduction:.2f}% < {profile.min_reduction_percent:.2f}%"
        )
    model.p5_internal_pruned_v22 = True
    model.pruning_metadata = {
        "profile": profile.name,
        "params_before": before,
        "params_after": after,
        "reduction_percent": reduction,
    }
    return model


def save_pruned_ultralytics_checkpoint(model: nn.Module, source_checkpoint: str | Path, destination: str | Path) -> Path:
    """Save a normal Ultralytics checkpoint and replace EMA with the pruned graph."""
    from ultralytics import YOLO

    source_checkpoint = Path(source_checkpoint)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    wrapper = YOLO(str(source_checkpoint))
    wrapper.model = model
    if isinstance(getattr(wrapper, "ckpt", None), dict):
        pruned_ema = copy.deepcopy(model).half()
        for parameter in pruned_ema.parameters():
            parameter.requires_grad_(False)
        wrapper.ckpt["ema"] = pruned_ema
        wrapper.ckpt["updates"] = 0
    wrapper.save(destination)
    return destination
