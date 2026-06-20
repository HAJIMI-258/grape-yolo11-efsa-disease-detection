from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True)
class P5PruningProfile:
    """Explicit output-channel targets for the YOLO11n P5 path."""

    name: str
    deep_channels: int
    p5_down_channels: int
    p5_head_channels: int
    expected_params: int


PROFILES = {
    # Approximate parameter counts are for the seven-class HighSource7 head.
    # Start with a very small cut because AP50 preservation is the hard constraint.
    "p251": P5PruningProfile("p251", 248, 128, 248, 2_506_905),
    "p242": P5PruningProfile("p242", 240, 128, 240, 2_424_749),
    "p233": P5PruningProfile("p233", 232, 120, 232, 2_333_649),
    "p226": P5PruningProfile("p226", 224, 120, 224, 2_255_845),
    "p210": P5PruningProfile("p210", 208, 112, 208, 2_095_773),
    "p196": P5PruningProfile("p196", 192, 112, 192, 1_955_301),
}


@dataclass(frozen=True)
class PruningTarget:
    """A safe output projection whose channel count can change without stale split attributes."""

    layer_index: int
    module_path: str
    target_channels: int


def _nested_module(module: nn.Module, path: str) -> nn.Module:
    current = module
    for token in path.split("."):
        current = getattr(current, token)
    return current


def _target_list(profile: P5PruningProfile) -> list[PruningTarget]:
    # Do not prune stem, P2/P3/P4 backbone, P3/P4 PAN outputs, regression heads,
    # classification heads, or hidden C2f/C2PSA dimensions. Only output projections
    # on the deep P5 route are changed, avoiding stale C2f `c` split attributes.
    return [
        PruningTarget(7, "conv", profile.deep_channels),
        PruningTarget(8, "cv2.conv", profile.deep_channels),
        PruningTarget(9, "cv2.conv", profile.deep_channels),
        PruningTarget(10, "cv2.conv", profile.deep_channels),
        PruningTarget(20, "conv", profile.p5_down_channels),
        PruningTarget(22, "cv2.conv", profile.p5_head_channels),
    ]


def _bn_scale_for_target(block: nn.Module, module_path: str, channels: int, device: torch.device) -> torch.Tensor:
    """Return the paired BN scale when the target convolution is wrapped by Ultralytics Conv."""
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

    # Rank channels by trained filter energy multiplied by the paired BN scale.
    # This preserves the learned channel subspace and is deliberately different
    # from the prefix slicing that failed in v17.
    weight_score = conv.weight.detach().float().flatten(1).pow(2).sum(1).sqrt()
    bn_scale = _bn_scale_for_target(block, module_path, conv.out_channels, weight_score.device)
    importance = weight_score * bn_scale
    return importance.argsort()[:prune_count].cpu().tolist()


def _refresh_metadata(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "np"):
            module.np = sum(parameter.numel() for parameter in module.parameters())
    head = model.model[-1]
    if hasattr(head, "shape"):
        head.shape = None
    if hasattr(head, "dynamic"):
        head.dynamic = True


def prune_yolo11n_p5(
    model: nn.Module,
    profile: P5PruningProfile,
    *,
    example_inputs: torch.Tensor | None = None,
) -> nn.Module:
    """Prune only the deep P5 route of a trained YOLO11n with DepGraph.

    Requires ``torch-pruning==1.6.1``. The dependency graph physically removes coupled BN channels and downstream input
    channels. Unlike hand-designed width scaling, P3/P4 features and both Detect towers retain their trained outputs.
    """
    try:
        import torch_pruning as tp
    except ImportError as exc:
        raise RuntimeError("Install the pinned dependency with: pip install -r requirements-pruning.txt") from exc

    model = model.float().cpu().eval()
    if example_inputs is None:
        example_inputs = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    example_inputs = example_inputs.to(next(model.parameters()).device)

    for target in _target_list(profile):
        block = model.model[target.layer_index]
        root = _nested_module(block, target.module_path)
        if not isinstance(root, nn.Conv2d):
            raise TypeError(f"Layer {target.layer_index}.{target.module_path} must be Conv2d, got {type(root)!r}")
        if root.out_channels < target.target_channels:
            raise ValueError(
                f"Target {target.target_channels} exceeds current channels {root.out_channels} "
                f"at layer {target.layer_index}.{target.module_path}"
            )
        prune_count = root.out_channels - target.target_channels
        if prune_count == 0:
            continue

        # Rebuild after every physical edit so subsequent dependencies and indices
        # reflect the current graph. Autograd must remain enabled for DepGraph.
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
    if output is None:
        raise RuntimeError("Pruned model forward returned None")

    actual_params = sum(parameter.numel() for parameter in model.parameters())
    tolerance = max(25_000, int(profile.expected_params * 0.03))
    if abs(actual_params - profile.expected_params) > tolerance:
        raise RuntimeError(
            f"Unexpected parameter count after {profile.name}: {actual_params:,}; "
            f"expected approximately {profile.expected_params:,}"
        )
    return model


def save_pruned_model(model: nn.Module, destination: str | Path, metadata: dict | None = None) -> Path:
    """Serialize the changed module graph; a state_dict alone cannot reconstruct it."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.zero_grad(set_to_none=True)
    torch.save(
        {
            "model": model.float().cpu(),
            "metadata": metadata or {},
        },
        destination,
    )
    return destination


def load_pruned_model(path: str | Path) -> nn.Module:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(model, nn.Module):
        raise TypeError(f"No serialized model graph found in {path}")
    return model.float()
