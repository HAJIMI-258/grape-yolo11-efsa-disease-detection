from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class LowRankConv(nn.Module):
    """SVD-factorized replacement for an Ultralytics Conv block.

    The block preserves the original output channels, stride, padding, dilation, batch normalization, and activation.
    Only the internal matrix rank changes.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        rank: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        act: nn.Module,
        bn: nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.rank = int(rank)
        self.spatial = nn.Conv2d(
            c1,
            rank,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            bias=False,
        )
        self.pointwise = nn.Conv2d(rank, c2, 1, 1, bias=False)
        self.bn = copy.deepcopy(bn)
        self.act = copy.deepcopy(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.spatial(x))))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


@dataclass(frozen=True)
class ConvGeometry:
    c1: int
    c2: int
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    act: nn.Module
    bn: nn.BatchNorm2d

    @property
    def flat_input(self) -> int:
        return self.c1 * self.kernel[0] * self.kernel[1]


@dataclass
class FactorCandidate:
    name: str
    layer_index: int
    module_path: str
    module: Conv | LowRankConv
    geometry: ConvGeometry
    matrix: torch.Tensor
    singular_values: torch.Tensor
    current_cost: int
    possible_ranks: list[int]
    sensitivity: float


@dataclass(frozen=True)
class RankChoice:
    name: str
    rank: int
    parameters_saved: int
    retained_energy: float


def _is_deep_relaxable_candidate(candidate: FactorCandidate) -> bool:
    """Allow a lower energy floor only on deep/context modules with low paper risk."""
    if candidate.layer_index in {7, 8, 10, 20, 22}:
        return True
    if candidate.layer_index == 9 and not candidate.module_path.startswith("cv2"):
        return True
    return False


def _energy_floor(candidate: FactorCandidate, base: float, deep_floor: float | None) -> float:
    """Return the retained-energy floor for one candidate.

    The global floor remains the default. A lower deep floor can be enabled for late compression stages, while known
    fragile context and fine-grained classification modules remain at a stricter floor.
    """
    floor = base
    if candidate.layer_index == 9 and candidate.module_path.startswith("cv2"):
        floor = max(floor, 0.985)
    if candidate.layer_index == 23 and (
        candidate.module_path.startswith("cv2.0") or candidate.module_path.startswith("cv2.1")
    ):
        floor = max(floor, 0.980)
    if deep_floor is not None and _is_deep_relaxable_candidate(candidate):
        floor = min(floor, deep_floor)
    return floor


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
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        return None
    return int(parts[1]), ".".join(parts[2:])


def _sensitivity(layer_index: int, path: str) -> float:
    """Protect shallow features and the P3/P4 classification towers."""
    if layer_index <= 4 or layer_index == 16:
        return float("inf")
    if layer_index == 23:
        if path.startswith("cv3.0") or path.startswith("cv3.1"):
            return float("inf")
        if path.startswith("cv2.0"):
            return 3.0
        if path.startswith("cv2.1") or path.startswith("cv3.2"):
            return 2.0
        return 1.5
    if layer_index in {7, 8, 9, 10, 20, 22}:
        return 1.0
    if layer_index in {5, 6, 13, 19}:
        return 2.0
    return 2.5


def _matrix_and_geometry(module: Conv | LowRankConv) -> tuple[torch.Tensor, ConvGeometry, int, int | None]:
    if isinstance(module, Conv):
        conv = module.conv
        if conv.groups != 1:
            raise ValueError("Grouped/depthwise convolutions are not eligible")
        matrix = conv.weight.detach().float().reshape(conv.out_channels, -1)
        geometry = ConvGeometry(
            c1=conv.in_channels,
            c2=conv.out_channels,
            kernel=tuple(conv.kernel_size),
            stride=tuple(conv.stride),
            padding=tuple(conv.padding),
            dilation=tuple(conv.dilation),
            act=module.act,
            bn=module.bn,
        )
        return matrix, geometry, conv.weight.numel(), None

    if isinstance(module, LowRankConv):
        spatial = module.spatial.weight.detach().float().reshape(module.rank, -1)
        pointwise = module.pointwise.weight.detach().float().reshape(module.pointwise.out_channels, module.rank)
        matrix = pointwise @ spatial
        conv = module.spatial
        geometry = ConvGeometry(
            c1=conv.in_channels,
            c2=module.pointwise.out_channels,
            kernel=tuple(conv.kernel_size),
            stride=tuple(conv.stride),
            padding=tuple(conv.padding),
            dilation=tuple(conv.dilation),
            act=module.act,
            bn=module.bn,
        )
        cost = module.spatial.weight.numel() + module.pointwise.weight.numel()
        return matrix, geometry, cost, module.rank

    raise TypeError(type(module))


def _factor_cost(geometry: ConvGeometry, rank: int) -> int:
    return rank * (geometry.flat_input + geometry.c2)


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
        if sensitivity == float("inf"):
            continue

        try:
            matrix, geometry, current_cost, current_rank = _matrix_and_geometry(module)
        except ValueError:
            continue
        if min(geometry.c1, geometry.c2) < 24 or current_cost < 4096:
            continue

        singular_values = torch.linalg.svdvals(matrix.cpu())
        max_matrix_rank = min(matrix.shape)
        if current_rank is None:
            break_even = (current_cost - 1) // (geometry.flat_input + geometry.c2)
            highest = min(max_matrix_rank, int(break_even))
        else:
            highest = min(max_matrix_rank, current_rank - rank_step)
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
                geometry=geometry,
                matrix=matrix.cpu(),
                singular_values=singular_values.cpu(),
                current_cost=current_cost,
                possible_ranks=possible,
                sensitivity=sensitivity,
            )
        )
    return candidates


def plan_lowrank_budget(
    model: nn.Module,
    target_parameters: int,
    rank_step: int = 8,
    min_retained_energy: float = 0.95,
    deep_min_retained_energy: float | None = None,
) -> list[RankChoice]:
    """Greedily trade the least singular-value energy for parameter savings."""
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
            next_cost = _factor_cost(candidate.geometry, next_rank)
            total_energy = candidate.singular_values.square().sum().clamp_min(1e-12)
            next_energy = float(candidate.singular_values[:next_rank].square().sum() / total_energy)
            candidate_floor = _energy_floor(candidate, min_retained_energy, deep_min_retained_energy)
            if next_energy < candidate_floor:
                continue

            if state_index is None:
                previous_cost = candidate.current_cost
                previous_energy = 1.0
            else:
                previous_rank = candidate.possible_ranks[state_index]
                previous_cost = _factor_cost(candidate.geometry, previous_rank)
                previous_energy = float(candidate.singular_values[:previous_rank].square().sum() / total_energy)

            saved = previous_cost - next_cost
            if saved <= 0:
                continue
            energy_loss = max(previous_energy - next_energy, 0.0)
            score = candidate.sensitivity * energy_loss / saved
            proposal = (score, candidate.name, next_index, saved)
            if best is None or proposal < best:
                best = proposal

        if best is None:
            floor_text = f"{min_retained_energy:.3f}"
            if deep_min_retained_energy is not None:
                floor_text += f", deep={deep_min_retained_energy:.3f}"
            raise RuntimeError(
                f"Unable to reach {target_parameters:,} parameters with the protected candidate set; "
                f"stopped near {predicted_total:,} with min_retained_energy={floor_text}"
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
        final_cost = _factor_cost(candidate.geometry, rank)
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
    matrix, geometry, _cost, _current_rank = _matrix_and_geometry(module)
    u, s, vh = torch.linalg.svd(matrix.cpu(), full_matrices=False)
    rank = min(rank, s.numel())
    replacement = LowRankConv(
        geometry.c1,
        geometry.c2,
        rank,
        geometry.kernel,
        geometry.stride,
        geometry.padding,
        geometry.dilation,
        geometry.act,
        geometry.bn,
    )
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
    min_retained_energy: float = 0.95,
    deep_min_retained_energy: float | None = None,
) -> tuple[nn.Module, list[RankChoice]]:
    """Plan, apply, and validate low-rank compression to a global budget."""
    model = model.float().cpu().eval()
    choices = plan_lowrank_budget(
        model,
        target_parameters,
        min_retained_energy=min_retained_energy,
        deep_min_retained_energy=deep_min_retained_energy,
    )
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
    """Save an Ultralytics checkpoint with EMA replaced by the low-rank graph."""
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
