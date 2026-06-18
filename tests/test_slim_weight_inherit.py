from __future__ import annotations

import torch
from torch import nn

from scripts.highsource7.slim_weight_inherit import inherit_narrow_model_weights


class _ToyDetector(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, bias=False), nn.BatchNorm2d(channels))
        self.cv3 = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Conv2d(channels, classes, 1))]
        )


def test_narrow_weight_inheritance_copies_common_channels_and_skips_wrong_class_head():
    source = _ToyDetector(channels=8, classes=80)
    target = _ToyDetector(channels=4, classes=7)
    for parameter in source.parameters():
        nn.init.constant_(parameter, 2.0)

    original_class_weight = target.cv3[0][1].weight.detach().clone()
    stats = inherit_narrow_model_weights(target, source, target_num_classes=7)

    assert stats.sliced_tensors > 0
    assert stats.skipped_tensors >= 2
    assert torch.allclose(target.stem[0].weight, torch.full_like(target.stem[0].weight, 2.0))
    assert torch.equal(target.cv3[0][1].weight.detach(), original_class_weight)


def test_narrow_weight_inheritance_keeps_matching_dataset_classifier():
    source = _ToyDetector(channels=8, classes=7)
    target = _ToyDetector(channels=4, classes=7)
    nn.init.constant_(source.cv3[0][1].weight, 3.0)

    inherit_narrow_model_weights(target, source, target_num_classes=7)

    assert torch.allclose(target.cv3[0][1].weight, torch.full_like(target.cv3[0][1].weight, 3.0))
