from __future__ import annotations

import copy

import torch.nn as nn

from ultralytics.nn.modules import Detect
from ultralytics.nn.modules.conv import Conv, DWConv


class FineGrainedDetect(Detect):
    """Detect-compatible head that widens only the classification towers.

    The box-regression towers, feature inputs, decoding path, and NMS behavior remain identical to YOLO11 Detect. The
    extra capacity is isolated to class discrimination, which is the repeated failure mode on HighSource7.
    """

    def __init__(
        self,
        nc: int = 80,
        cls_channels: int = 96,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
        legacy: bool = False,
    ):
        self.legacy = bool(legacy)
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c3 = max(int(cls_channels), min(self.nc, 100))
        self.cls_channels = c3
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        if end2end:
            self.one2one_cv3 = copy.deepcopy(self.cv3)


def replace_detect_with_finegrained(model, cls_channels: int = 96):
    """Replace a model's final Detect module without changing its feature topology."""
    old = model.model[-1]
    if not isinstance(old, Detect):
        raise TypeError(f"Expected final Detect module, got {type(old)!r}")

    channels = tuple(int(branch[0].conv.in_channels) for branch in old.cv2)
    new = FineGrainedDetect(
        nc=old.nc,
        cls_channels=cls_channels,
        reg_max=old.reg_max,
        end2end=old.end2end,
        ch=channels,
        legacy=old.legacy,
    )

    # Keep regression weights and all runtime geometry from the original head.
    new.cv2 = old.cv2
    new.dfl = old.dfl
    new.stride = old.stride.clone()
    new.anchors = old.anchors
    new.strides = old.strides
    if old.end2end:
        new.one2one_cv2 = old.one2one_cv2

    # Initialize the widened classification towers after copying the parsed
    # strides; Detect.bias_init() uses them for finite class priors.
    new.bias_init()

    # Ultralytics attaches graph metadata after YAML parsing. Preserve it when
    # swapping the head post-construction so DetectionModel.forward can route
    # inputs exactly as before.
    for attr in ("i", "f"):
        if hasattr(old, attr):
            setattr(new, attr, getattr(old, attr))
    new.type = f"{new.__module__}.{new.__class__.__name__}"
    new.np = sum(parameter.numel() for parameter in new.parameters())

    model.model[-1] = new
    model.stride = new.stride
    return model
