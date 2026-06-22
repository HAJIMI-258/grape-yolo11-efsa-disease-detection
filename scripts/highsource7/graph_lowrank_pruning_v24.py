from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from lowrank_compression_v23 import LowRankConv


@dataclass(frozen=True)
class GraphRankChoice:
    """One applied dependency-graph pruning step inside a LowRankConv rank space."""

    name: str
    old_rank: int
    new_rank: int
    parameters_saved: int
    retained_importance: float
    pruned_indices: tuple[int, ...]


@dataclass(frozen=True)
class _Candidate:
    name: str
    layer_index: int
    module_path: str
    module: LowRankConv
    sensitivity: float


def _top_layer_and_path(name: str) -> tuple[int, str] | None:
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        return None
    return int(parts[1]), ".".join(parts[2:])


def _candidate_sensitivity(layer_index: int, path: str) -> float:
    """Lower values are easier to prune.

    V24 keeps lesion-sensitive shallow layers and P3/P4 class towers protected, while allowing deeper context and P5
    low-rank interaction channels to shrink first.
    """
    if layer_index <= 4 or layer_index == 16:
        return float("inf")
    if layer_index == 23:
        if path.startswith("cv3.0") or path.startswith("cv3.1"):
            return float("inf")
        if path.startswith("cv2.0") or path.startswith("cv2.1"):
            return 3.0
        if path.startswith("cv2.2") or path.startswith("cv3.2"):
            return 1.0
        return 2.0
    if layer_index in {7, 8, 10, 20, 22}:
        return 1.0
    if layer_index == 9:
        return 2.5 if path.startswith("cv2") else 1.5
    if layer_index in {5, 6, 13, 19}:
        return 2.5
    return 3.0


def _eligible_candidates(model: nn.Module) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for name, module in model.named_modules():
        if not isinstance(module, LowRankConv):
            continue
        parsed = _top_layer_and_path(name)
        if parsed is None:
            continue
        layer_index, module_path = parsed
        sensitivity = _candidate_sensitivity(layer_index, module_path)
        if sensitivity == float("inf"):
            continue
        if module.spatial.out_channels <= 8:
            continue
        candidates.append(_Candidate(name, layer_index, module_path, module, sensitivity))
    return candidates


def _rank_importance(module: LowRankConv) -> torch.Tensor:
    """Score internal rank channels by both sides of the factorized interaction.

    The spatial filter norm measures how much input evidence a rank channel extracts. The pointwise column norm measures
    how strongly that rank channel contributes back to the external output channels. If gradients are present, a small
    Taylor/Fisher proxy is added without requiring a separate code path.
    """
    spatial = module.spatial.weight.detach().float().flatten(1)
    pointwise = module.pointwise.weight.detach().float().flatten(0, 1).view(module.pointwise.out_channels, -1)
    score = spatial.pow(2).sum(1).sqrt() * pointwise.pow(2).sum(0).sqrt()

    if module.spatial.weight.grad is not None and module.pointwise.weight.grad is not None:
        spatial_grad = module.spatial.weight.grad.detach().float().flatten(1)
        pointwise_grad = (
            module.pointwise.weight.grad.detach().float().flatten(0, 1).view(module.pointwise.out_channels, -1)
        )
        fisher = (spatial * spatial_grad).pow(2).sum(1) + (pointwise * pointwise_grad).pow(2).sum(0)
        score = score + fisher.sqrt()
    return score.clamp_min(1e-12)


def _current_rank(module: LowRankConv) -> int:
    return int(module.spatial.out_channels)


def _refresh_lowrank_metadata(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, LowRankConv):
            module.rank = int(module.spatial.out_channels)
        if hasattr(module, "np"):
            module.np = sum(parameter.numel() for parameter in module.parameters())
    head = model.model[-1]
    if hasattr(head, "shape"):
        head.shape = None
    if hasattr(head, "dynamic"):
        head.dynamic = True


def _validate_three_scale_output(model: nn.Module, example_inputs: torch.Tensor) -> None:
    with torch.no_grad():
        output = model(example_inputs)
    raw = output[1] if isinstance(output, tuple) else output
    if not isinstance(raw, dict) or raw["scores"].shape[-1] != 8400:
        raise RuntimeError("V24 must preserve all three scales and 8400 anchors")


def graph_prune_lowrank_ranks(
    model: nn.Module,
    target_parameters: int,
    *,
    example_inputs: torch.Tensor | None = None,
    prune_step: int = 4,
    min_rank: int = 8,
    min_retained_importance: float = 0.92,
    max_steps: int = 512,
) -> tuple[nn.Module, list[GraphRankChoice]]:
    """Prune internal LowRankConv rank channels with Torch-Pruning dependency groups.

    This is graph compression, not another SVD pass: the removed channels are physically deleted from the spatial factor
    and the coupled input dimension of the pointwise factor. External feature widths and Detect outputs are preserved.
    """
    try:
        import torch_pruning as tp
    except ImportError as exc:
        raise RuntimeError("Install the pinned dependency with: pip install -r requirements-pruning.txt") from exc

    if prune_step <= 0:
        raise ValueError(f"prune_step must be positive, got {prune_step}")
    if min_rank <= 0:
        raise ValueError(f"min_rank must be positive, got {min_rank}")

    model = model.float().cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if example_inputs is None:
        example_inputs = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    example_inputs = example_inputs.to(next(model.parameters()).device)
    _validate_three_scale_output(model, example_inputs)

    choices: list[GraphRankChoice] = []
    for _ in range(max_steps):
        current_total = sum(parameter.numel() for parameter in model.parameters())
        if current_total <= target_parameters:
            break

        best: tuple[float, _Candidate, list[int], float, int] | None = None
        for candidate in _eligible_candidates(model):
            rank = _current_rank(candidate.module)
            step = min(prune_step, rank - min_rank)
            if step <= 0:
                continue
            scores = _rank_importance(candidate.module)
            if scores.numel() != rank:
                continue
            prune_indices = scores.argsort()[:step].cpu().tolist()
            total_score = scores.sum().clamp_min(1e-12)
            retained = float((total_score - scores[prune_indices].sum()) / total_score)
            if retained < min_retained_importance:
                continue

            saved = step * (candidate.module.spatial.weight[0].numel() + candidate.module.pointwise.out_channels)
            importance_loss = max(1.0 - retained, 0.0)
            score = candidate.sensitivity * importance_loss / max(saved, 1)
            proposal = (score, candidate, prune_indices, retained, saved)
            if best is None or proposal[0] < best[0]:
                best = proposal

        if best is None:
            raise RuntimeError(
                f"Unable to reach {target_parameters:,} parameters with V24 protected rank pruning; "
                f"stopped near {current_total:,}"
            )

        _score, candidate, prune_indices, retained, predicted_saved = best
        old_rank = _current_rank(candidate.module)
        with torch.enable_grad():
            graph = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
        group = graph.get_pruning_group(candidate.module.spatial, tp.prune_conv_out_channels, idxs=prune_indices)
        if not graph.check_pruning_group(group):
            raise RuntimeError(f"Invalid V24 pruning group at {candidate.name}: {group}")
        group.prune()
        _refresh_lowrank_metadata(model)
        new_rank = _current_rank(candidate.module)
        choices.append(
            GraphRankChoice(
                name=candidate.name,
                old_rank=old_rank,
                new_rank=new_rank,
                parameters_saved=predicted_saved,
                retained_importance=retained,
                pruned_indices=tuple(int(index) for index in prune_indices),
            )
        )

    _refresh_lowrank_metadata(model)
    _validate_three_scale_output(model, example_inputs)
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual > target_parameters:
        raise RuntimeError(f"V24 graph-pruned model exceeds target: {actual:,} > {target_parameters:,}")
    model.graph_lowrank_v24 = True
    model.graph_lowrank_metadata = {
        "target_parameters": target_parameters,
        "actual_parameters": actual,
        "choices": [choice.__dict__ for choice in choices],
    }
    return model, choices


def save_graph_lowrank_checkpoint(model: nn.Module, source_checkpoint: str | Path, destination: str | Path) -> Path:
    """Save an Ultralytics checkpoint with EMA replaced by the V24 graph-pruned model."""
    from ultralytics import YOLO

    source_checkpoint = Path(source_checkpoint)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapper = YOLO(str(source_checkpoint))
    wrapper.model = model
    if isinstance(getattr(wrapper, "ckpt", None), dict):
        ema = copy.deepcopy(model).half()
        for parameter in ema.parameters():
            parameter.requires_grad_(False)
        wrapper.ckpt["ema"] = ema
        wrapper.ckpt["updates"] = 0
    wrapper.save(destination)
    return destination
