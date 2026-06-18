from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class SlimInheritanceStats:
    """Summary of overlap-based weight inheritance into a narrower model."""

    exact_tensors: int
    sliced_tensors: int
    skipped_tensors: int
    copied_elements: int
    target_elements: int
    skipped_keys: tuple[str, ...]

    @property
    def element_coverage(self) -> float:
        """Return the fraction of target state elements initialized from the source model."""
        return self.copied_elements / max(self.target_elements, 1)


def _is_dynamic_detection_buffer(key: str) -> bool:
    """Skip runtime-generated detection buffers that should be rebuilt by the target model."""
    suffixes = (".anchors", ".strides", ".stride")
    return key.endswith(suffixes)


def _is_incompatible_classifier_output(
    key: str,
    source: torch.Tensor,
    target: torch.Tensor,
    target_num_classes: int | None,
) -> bool:
    """Avoid copying final class logits when source and target class counts differ."""
    if target_num_classes is None or target.ndim == 0 or source.ndim == 0:
        return False
    is_cls_head = (
        key.startswith("cv3.")
        or ".cv3." in key
        or key.startswith("one2one_cv3.")
        or ".one2one_cv3." in key
    )
    if not is_cls_head:
        return False
    return target.shape[0] == target_num_classes and source.shape[0] != target.shape[0]


def _compatible_for_overlap(source: torch.Tensor, target: torch.Tensor) -> bool:
    """Return whether two tensors can be copied by taking a prefix on channel dimensions."""
    if source.ndim != target.ndim:
        return False
    if source.ndim == 0:
        return True
    # Spatial kernels must match. Channel dimensions may differ.
    if source.ndim >= 3 and source.shape[-2:] != target.shape[-2:]:
        return False
    return True


def _copy_prefix(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Copy the largest common prefix from source into a clone of target."""
    output = target.detach().clone()
    if source.ndim == 0:
        output.copy_(source.to(device=output.device, dtype=output.dtype))
        return output, 1

    common_shape = tuple(min(int(s), int(t)) for s, t in zip(source.shape, target.shape))
    if any(size <= 0 for size in common_shape):
        return output, 0
    slices = tuple(slice(0, size) for size in common_shape)
    output[slices].copy_(source[slices].to(device=output.device, dtype=output.dtype))
    copied = 1
    for size in common_shape:
        copied *= size
    return output, copied


def inherit_narrow_model_weights(
    target_model: nn.Module,
    source_model: nn.Module,
    *,
    target_num_classes: int | None = None,
    ignored_keys: Iterable[str] = (),
) -> SlimInheritanceStats:
    """Initialize a topology-compatible narrow model from a wider checkpoint.

    The HighSource7 width-0.20 student keeps the YOLO11n layer topology but uses
    fewer channels. Standard ``load_state_dict`` rejects most useful tensors
    because their channel dimensions differ. This routine copies the common
    channel prefix for Conv/BN/head tensors while preserving target-only values.

    The method is intended for structured compression experiments. When the
    source is a dataset-trained baseline, report that initialization explicitly;
    it is not a same-start architecture comparison.
    """
    source_state = source_model.float().state_dict()
    target_state = target_model.state_dict()
    ignored = set(ignored_keys)

    exact_tensors = 0
    sliced_tensors = 0
    skipped_tensors = 0
    copied_elements = 0
    target_elements = sum(int(t.numel()) for t in target_state.values())
    skipped_keys: list[str] = []

    inherited_state: dict[str, torch.Tensor] = {}
    for key, target in target_state.items():
        source = source_state.get(key)
        if (
            source is None
            or key in ignored
            or _is_dynamic_detection_buffer(key)
            or _is_incompatible_classifier_output(key, source, target, target_num_classes)
            or not _compatible_for_overlap(source, target)
        ):
            inherited_state[key] = target
            skipped_tensors += 1
            skipped_keys.append(key)
            continue

        copied_tensor, copied = _copy_prefix(source, target)
        if copied == 0:
            inherited_state[key] = target
            skipped_tensors += 1
            skipped_keys.append(key)
            continue

        inherited_state[key] = copied_tensor
        copied_elements += copied
        if source.shape == target.shape:
            exact_tensors += 1
        else:
            sliced_tensors += 1

    target_model.load_state_dict(inherited_state, strict=True)
    return SlimInheritanceStats(
        exact_tensors=exact_tensors,
        sliced_tensors=sliced_tensors,
        skipped_tensors=skipped_tensors,
        copied_elements=copied_elements,
        target_elements=target_elements,
        skipped_keys=tuple(skipped_keys),
    )
