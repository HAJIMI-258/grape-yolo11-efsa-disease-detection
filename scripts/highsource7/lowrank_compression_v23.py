from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, autopad


class LowRankConv(nn.Module):
    """SVD-factorized replacement for an Ultralytics Conv block.

    Output channels, stride, padding, dilation, BN, and activation are preserved.
    Only the internal matrix rank changes, so the YOLO graph interface is stable.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        rank: int,
        k: int | tuple[int, int] = 1,
        s: int | tuple[int, int] = 1,
        p: int | tuple[int, int] | None = None,
        d: int | tuple[int, int] = 1,
        act: nn.Module | None = None,
        bn: nn.BatchNorm2d | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.rank = int(rank)
        self.spatial = nn.Conv2d(c1, rank, k, s, autopad(k, p, d), dilation=d, bias=False)
        self.pointwise = nn.Conv2d(rank, c2, 1, 1, bias=False)
        self.bn = copy.deepcopy(bn) if bn is not None else nn.BatchNorm2d(c2)
        self.act = copy.deepcopy(act) if act is not None else nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.spatial(x))))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


@dataclass
class FactorCandidate:
    name: str
    layer_index: int
    module_path: str
    module: Conv | LowRankConv
    matrix: torch.Tensor
    singular_values: torch.Tensor
    current_cost: int
    current_rank: int | None
    possible_ranks: list[int]
    sensitivity: float


@dataclass(frozen=True)
class RankChoice:
    name: str
    rank: int
    parameters_saved: int
    retained_energy: float


def _nested_module(module: nn.Module, path: str) -> nn.Module:
    current = module
    if not path:
        return current
    for token in path.split("."):
        if token.isdigit() and isinstance(current, (nn.ModuleList, nn.Sequential)):
            current = current[int(token)]
        else:
            current = getattr(current, token)
    return current


def _set_nested_module(module: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = _nested_module(module, parent_path) if parent_path else module
    if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def _top_layer_and_path(name: str) -> tuple[int, str] | None:
    # DetectionModel named modules use names such as model.8.m.0.cv1.
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        return None
    return int(parts[1]), ".".join(parts[2:])


def _sensitivity(layer_index: int, path: str) -> float:
    """Penalize approximation of lesion-sensitive P3/P4 classification paths."""
    if layer_index <= 4:
        return float("inf")
    if layer_index == 16:  # P3 PAN output
        return float("inf")
    if layer_index == 23:
        if path.startswith("cv3.0") or path.startswith("cv3.1"):
            return float("inf")  # preserve P3/P4 classification towers
        if path.startswith("cv2.0"):
            return 3.0
        if path.startswith("cv2.1") or path.startswith("cv3.2"):
            return 2.0
        return 1.5
    if layer_index in {5, 6, 13, 19}:
        return 2.0
    if layer_index in {7, 8, 9, 10, 20, 22}:
        return 1.0
    return 2.5


def _matrix_from_module(module: Conv | LowRankConv) -> tuple[torch.Tensor, int, int, tuple, tuple, tuple, nn.Module, nn.BatchNorm2d]:
    if isinstance(module, Conv):
        conv = module.conv
        if conv.groups != 1:
            raise ValueError("Grouped/depthwise convolutions are not eligible")
        matrix = conv.weight.detach().float().reshape(conv.out_channels, -1)
        return matrix, conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.dilation, module.act, module.bn

    if isinstance(module, LowRankConv):
        spatial = module.spatial.weight.detach().float().reshape(module.rank, -1)
        pointwise = module.pointwise.weight.detach().float().reshape(module.pointwise.out_channels, module.rank)
        matrix = pointwise @ spatial
        conv = module.spatial
        return matrix, conv.in_channels, module.pointwise.out_channels, conv.kernel_size, conv.stride, conv.dilation, module.act, module.bn

    raise TypeError(type(module))


def _parameter_cost(c1: int, c2: int, kernel_size: tuple[int, int], rank: int) -> int:
    flat_input = c1 * kernel_size[0] * kernel_size[1]
    return rank * (flat_input + c2)


def _candidate_modules(model: nn.Module, rank_step: int = 8, min_rank: int = 16) -> list[FactorCandidate]:
    candidates: list[FactorCandidate] = []
    for name, module in model.named_modules():
        if not isinstance(module, (Conv, LowRankConv)):
            continue
        parsed = _top_layer_and_path(name)
        if parsed is None:
            continue
        layer_index, module_path = parsed
        sensitivity = _sensitivity(layer_index, module_path)
        if not torch.isfinite(torch.tensor(sensitivity)):
            continue

        if isinstance(module, Conv):
            conv = module.conv
            if conv.groups != 1 or min(conv.in_channels, conv.out_channels) < 24:
                continue
            current_cost = conv.weight.numel()
            current_rank = None
        else:
            conv = module.spatial
            if conv.groups != 1:
                continue
            current_cost = module.spatial.weight.numel() + module.pointwise.weight.numel()
            current_rank = module.rank

        if current_cost < 4096:
            continue

        matrix, c1, c2, kernel, _stride, _dilation, _act, _bn = _matrix_from_module(module)
        singular_values = torch.linalg.svdvals(matrix.cpu())
        max_rank = min(matrix.shape)
        if current_rank is None:
            break_even = (current_cost - 1) // (c1 * kernel[0] * kernel[1] + c2)
            highest = min(max_rank, int(break_even))
        else:
            highest = min(max_rank, current_rank - rank_step)
        highest = highest // rank_step * rank_step
        floor = max(min_rank, rank_step)
        possible = list(range(highest, floor - 1, -rank_step)) if highest >= floor else []
        if not possible:
            continue

        candidates.append(
            FactorCandidate(
                name=name,
                layer_index=layer_index,
                module_path=module_path,
                module=module,
                matrix=matrix.cpu(),
                singular_values=singular_values.cpu(),
                current_cost=current_cost,
                current_rank=current_rank,
                possible_ranks=possible,
                sensitivity=sensitivity,
            )
        )
    return candidates


def plan_lowrank_budget(model: nn.Module, target_parameters: int, rank_step: int = 8) -> list[RankChoice]:
    """Greedily spend singular-value energy to reach a global parameter budget."""
    current_total = sum(parameter.numel() for parameter in model.parameters())
    if current_total <= target_parameters:
        return []

    candidates = _candidate_modules(model, rank_step=rank_step)
    if not candidates:
        raise RuntimeError("No eligible Conv modules were found for low-rank compression")

    states: dict[str, int | None] = {candidate.name: None for candidate in candidates}
    candidate_map = {candidate.name: candidate for candidate in candidates}
    predicted_total = current_total

    while predicted_total > target_parameters:
        best: tuple[float, str, int, int] | None = None
        for candidate in candidates:
            state_index = states[candidate.name]
            next_index = 0 if state_index is None else state_index + 1
            if next_index >= len(candidate.possible_ranks):
                continue
            next_rank = candidate.possible_ranks[next_index]
            c1 = candidate.matrix.shape[1] // (
                candidate.module.conv.kernel_size[0] * candidate.module.conv.kernel_size[1]
                if isinstance(candidate.module, Conv)
                else candidate.module.spatial.kernel_size[0] * candidate.module.spatial.kernel_size[1]
            )
            c2 = candidate.matrix.shape[0]
            kernel = candidate.module.conv.kernel_size if isinstance(candidate.module, Conv) else candidate.module.spatial.kernel_size
            next_cost = _parameter_cost(c1, c2, kernel, next_rank)
            if state_index is None:
                previous_cost = candidate.current_cost
                previous_energy = 1.0
            else:
                previous_rank = candidate.possible_ranks[state_index]
                previous_cost = _parameter_cost(c1, c2, kernel, previous_rank)
                total_energy = candidate.singular_values.square().sum().clamp_min(1e-12)
                previous_energy = float(candidate.singular_values[:previous_rank].square().sum() / total_energy)
            saved = previous_cost - next_cost
            if saved <= 0:
                continue
            total_energy = candidate.singular_values.square().sum().clamp_min(1e-12)
            next_energy = float(candidate.singular_values[:next_rank].square().sum() / total_energy)
            energy_loss = max(previous_energy - next_energy, 0.0)
            score = candidate.sensitivity * energy_loss / saved
            proposal = (score, candidate.name, next_index, saved)
            if best is None or proposal < best:
                best = proposal

        if best is None:
            raise RuntimeError(
                f"Unable to reach {target_parameters:,} parameters with the protected low-rank candidate set; "
                f"stopped at approximately {predicted_total:,}"
            )
        _, name, next_index, saved = best
        states[name] = next_index
        predicted_total -= saved

    choices: list[RankChoice] = []
    for name, index in states.items():
        if index is None:
            continue
        candidate = candidate_map[name]
        rank = candidate.possible_ranks[index]
        c1 = candidate.matrix.shape[1] // (
            candidate.module.conv.kernel_size[0] * candidate.module.conv.kernel_size[1]
            if isinstance(candidate.module, Conv)
            else candidate.module.spatial.kernel_size[0] * candidate.module.spatial.kernel_size[1]
        )
        c2 = candidate.matrix.shape[0]
        kernel = candidate.module.conv.kernel_size if isinstance(candidate.module, Conv) else candidate.module.spatial.kernel_size
        final_cost = _parameter_cost(c1, c2, kernel, rank)
        energy = candidate.singular_values.square().sum().clamp_min(1e-12)
        retained = float(candidate.singular_values[:rank].square().sum() / energy)
        choices.append(
            RankChoice(
                name=name,
                rank=rank,
                parameters_saved=candidate.current_cost - final_cost,
                retained_energy=retained,
            )
        )
    return sorted(choices, key=lambda item: item.name)


def _factorized_module(module: Conv | LowRankConv, rank: int) -> LowRankConv:
    matrix, c1, c2, kernel, stride, dilation, act, bn = _matrix_from_module(module)
    u, s, vh = torch.linalg.svd(matrix.cpu(), full_matrices=False)
    rank = min(rank, s.numel())
    replacement = LowRankConv(c1, c2, rank, kernel, stride, d=dilation, act=act, bn=bn)
    replacement.spatial.weight.data.copy_(vh[:rank].reshape_as(replacement.spatial.weight))
    replacement.pointwise.weight.data.copy_((u[:, :rank] * s[:rank]).reshape_as(replacement.pointwise.weight))
    return replacement


def apply_lowrank_plan(model: nn.Module, choices: list[RankChoice]) -> nn.Module:
    choice_map = {choice.name: choice for choice in choices}
    for name, module in list(model.named_modules()):
        choice = choice_map.get(name)
        if choice is None:
            continue
        parsed = _top_layer_and_path(name)
        if parsed is None:
            raise RuntimeError(f"Cannot locate chosen module {name}")
        layer_index, module_path = parsed
        replacement = _factorized_module(module, choice.rank)
        _set_nested_module(model.model[layer_index], module_path, replacement)

    for module in model.modules():
        if hasattr(module, "np"):
            module.np = sum(parameter.numel() for parameter in module.parameters())
    head = model.model[-1]
    head.shape = None
    head.dynamic = True
    return model


def compress_to_budget(
    model: nn.Module,
    target_parameters: int,
    *,
    example_inputs: torch.Tensor | None = None,
) -> tuple[nn.Module, list[RankChoice]]:
    """Plan, apply, and validate low-rank compression to an exact global budget."""
    model = model.float().cpu().eval()
    choices = plan_lowrank_budget(model, target_parameters)
    model = apply_lowrank_plan(model, choices)
    if example_inputs is None:
        example_inputs = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        output = model(example_inputs)
    raw = output[1] if isinstance(output, tuple) else output
    if not isinstance(raw, dict) or raw["scores"].shape[-1] != 8400:
        raise RuntimeError("V23 must preserve all three scales and 8400 anchors")
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual > target_parameters:
        raise RuntimeError(f"Low-rank model exceeds target: {actual:,} > {target_parameters:,}")
    model.lowrank_v23 = True
    model.lowrank_metadata = {
        "target_parameters": target_parameters,
        "actual_parameters": actual,
        "choices": [choice.__dict__ for choice in choices],
    }
    return model, choices


def save_lowrank_checkpoint(model: nn.Module, source_checkpoint: str | Path, destination: str | Path) -> Path:
    """Save a normal Ultralytics checkpoint with EMA replaced by the low-rank graph."""
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
