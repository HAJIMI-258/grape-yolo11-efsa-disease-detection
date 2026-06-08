# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Grape-EFSA modules for YOLO11 grape disease detection.

These modules migrate the feature-enhancement ideas from the Grape-EFSA-RTDETR method into the YOLO11
backbone/neck while keeping the native YOLO Detect head and loss path unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("EFSAEnhance", "CARAFEUp", "BiFPNFuse", "CoordECA")


def _inverse_tanh_clamped(value: float) -> float:
    """Return atanh(value) with numerical safety."""
    value = max(min(value, 0.999), -0.999)
    return 0.5 * math.log((1.0 + value) / (1.0 - value))


class SafeResidualScale(nn.Module):
    """Bounded residual scaling used by all EFSA enhancement modules."""

    def __init__(self, init_scale: float = 0.015, max_scale: float = 0.10, trainable: bool = True):
        """Initialize a safe residual scale.

        Args:
            init_scale (float): Initial absolute residual scale.
            max_scale (float): Maximum absolute residual scale after tanh bounding.
            trainable (bool): Whether the raw scale is trainable.
        """
        super().__init__()
        self.max_scale = float(max_scale)
        raw = _inverse_tanh_clamped(float(init_scale) / max(self.max_scale, 1e-6))
        tensor = torch.tensor(raw, dtype=torch.float32)
        if trainable:
            self.raw_scale = nn.Parameter(tensor)
        else:
            self.register_buffer("raw_scale", tensor)

    def forward(self) -> torch.Tensor:
        """Return the bounded residual scale."""
        return self.max_scale * torch.tanh(self.raw_scale)


class _DepthwiseBranch(nn.Module):
    """Depthwise convolution branch with BN and SiLU."""

    def __init__(self, channels: int, kernel_size: int):
        """Initialize a depthwise branch."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the branch."""
        return self.block(x)


class EFSAEnhance(nn.Module):
    """Edge-guided fine-grained small-lesion enhancement for YOLO11 feature maps.

    The module keeps input and output shape identical. It combines a fixed Prewitt edge/high-frequency branch,
    multi-kernel depthwise lesion texture branches, and local output refinement, then injects them through a bounded
    residual gate.
    """

    def __init__(
        self,
        c1: int,
        edge_alpha: float = 0.12,
        lesion_alpha: float = 0.10,
        refine_alpha: float = 0.08,
        init_scale: float = 0.015,
        max_scale: float = 0.10,
        trainable_scale: bool = True,
    ):
        """Initialize EFSAEnhance.

        Args:
            c1 (int): Number of input/output channels.
            edge_alpha (float): Weight for Prewitt edge/high-frequency branch.
            lesion_alpha (float): Weight for multi-scale small-lesion branch.
            refine_alpha (float): Weight for output refine branch.
            init_scale (float): Initial bounded residual scale.
            max_scale (float): Maximum residual scale.
            trainable_scale (bool): Whether the residual scale is trainable.
        """
        super().__init__()
        self.edge_alpha = float(edge_alpha)
        self.lesion_alpha = float(lesion_alpha)
        self.refine_alpha = float(refine_alpha)
        self.scale = SafeResidualScale(init_scale, max_scale, trainable_scale)

        prewitt_x = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 3.0
        prewitt_y = torch.tensor([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).view(1, 1, 3, 3) / 3.0
        self.register_buffer("prewitt_x", prewitt_x)
        self.register_buffer("prewitt_y", prewitt_y)
        self.edge_project = Conv(c1, c1, 1)

        self.lesion_3 = _DepthwiseBranch(c1, 3)
        self.lesion_5 = _DepthwiseBranch(c1, 5)
        self.lesion_7 = _DepthwiseBranch(c1, 7)
        self.lesion_fuse = Conv(c1 * 3, c1, 1)

        self.refine = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply EFSA enhancement."""
        channels = x.shape[1]
        wx = self.prewitt_x.repeat(channels, 1, 1, 1)
        wy = self.prewitt_y.repeat(channels, 1, 1, 1)
        gx = F.conv2d(x, wx, padding=1, groups=channels)
        gy = F.conv2d(x, wy, padding=1, groups=channels)
        edge = torch.sqrt(gx.square() + gy.square() + 1e-6)
        high_freq = x - F.avg_pool2d(x, 3, stride=1, padding=1)
        edge_out = self.edge_project(edge + high_freq)

        lesion = self.lesion_fuse(torch.cat((self.lesion_3(x), self.lesion_5(x), self.lesion_7(x)), dim=1))
        refine = self.refine(x)
        enhancement = self.edge_alpha * edge_out + self.lesion_alpha * lesion + self.refine_alpha * refine
        return x + self.scale() * enhancement


class CARAFEUp(nn.Module):
    """Pure PyTorch content-aware upsampling used as a lightweight CARAFE-style replacement."""

    def __init__(self, c1: int, scale_factor: int = 2, kernel_size: int = 3, compressed_channels: int = 32):
        """Initialize CARAFEUp.

        Args:
            c1 (int): Number of input/output channels.
            scale_factor (int): Upsampling factor.
            kernel_size (int): Reassembly kernel size.
            compressed_channels (int): Channels used by the kernel encoder.
        """
        super().__init__()
        if scale_factor < 1:
            raise ValueError("scale_factor must be >= 1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.c1 = int(c1)
        self.scale_factor = int(scale_factor)
        self.kernel_size = int(kernel_size)
        hidden = max(1, min(int(compressed_channels), self.c1))
        self.compress = Conv(self.c1, hidden, 1)
        self.encoder = nn.Conv2d(
            hidden,
            self.kernel_size * self.kernel_size * self.scale_factor * self.scale_factor,
            self.kernel_size,
            padding=self.kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample a feature map with content-aware reassembly weights."""
        if self.scale_factor == 1:
            return x
        b, c, h, w = x.shape
        out_h, out_w = h * self.scale_factor, w * self.scale_factor
        kernel_area = self.kernel_size * self.kernel_size
        weights = self.encoder(self.compress(x))
        weights = F.pixel_shuffle(weights, self.scale_factor)
        weights = weights.view(b, kernel_area, out_h, out_w).softmax(dim=1)

        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        patches = patches.view(b, c * kernel_area, h, w)
        patches = F.interpolate(patches, size=(out_h, out_w), mode="nearest")
        patches = patches.view(b, c, kernel_area, out_h, out_w)
        out = (patches * weights.unsqueeze(1)).sum(dim=2)
        if out.shape[-2:] != (out_h, out_w):
            raise RuntimeError(f"CARAFEUp output shape {out.shape[-2:]} != expected {(out_h, out_w)}")
        return out


class BiFPNFuse(nn.Module):
    """BiFPN-style normalized weighted fusion for multiple YOLO feature maps."""

    def __init__(self, c1_list: list[int], c2: int, eps: float = 1e-4):
        """Initialize BiFPNFuse.

        Args:
            c1_list (list[int]): Input channels for each incoming feature.
            c2 (int): Output channels.
            eps (float): Numerical epsilon for weight normalization.
        """
        super().__init__()
        if isinstance(c1_list, int):
            c1_list = [c1_list]
        self.eps = float(eps)
        self.proj = nn.ModuleList(Conv(int(c1), int(c2), 1) for c1 in c1_list)
        self.weights = nn.Parameter(torch.ones(len(c1_list), dtype=torch.float32))

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
        """Fuse input tensors after channel projection and spatial alignment."""
        if isinstance(xs, torch.Tensor):
            xs = [xs]
        target_size = xs[0].shape[-2:]
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.eps)
        fused = None
        for x, proj, weight in zip(xs, self.proj, weights):
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="nearest")
            y = weight * proj(x)
            fused = y if fused is None else fused + y
        return fused


class CoordECA(nn.Module):
    """Coordinate attention plus ECA channel attention with safe residual injection."""

    def __init__(
        self,
        c1: int,
        coord_alpha: float = 0.10,
        eca_alpha: float = 0.08,
        reduction: int = 32,
        eca_kernel_size: int = 3,
        init_scale: float = 0.015,
        max_scale: float = 0.10,
        trainable_scale: bool = True,
    ):
        """Initialize CoordECA.

        Args:
            c1 (int): Number of input/output channels.
            coord_alpha (float): Coordinate attention branch weight.
            eca_alpha (float): ECA branch weight.
            reduction (int): Coordinate attention reduction ratio.
            eca_kernel_size (int): ECA 1D convolution kernel.
            init_scale (float): Initial bounded residual scale.
            max_scale (float): Maximum residual scale.
            trainable_scale (bool): Whether the residual scale is trainable.
        """
        super().__init__()
        self.coord_alpha = float(coord_alpha)
        self.eca_alpha = float(eca_alpha)
        hidden = max(8, int(c1) // int(reduction))
        self.coord_conv = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.coord_h = nn.Conv2d(hidden, c1, 1)
        self.coord_w = nn.Conv2d(hidden, c1, 1)
        self.eca = nn.Conv1d(1, 1, kernel_size=eca_kernel_size, padding=eca_kernel_size // 2, bias=False)
        self.scale = SafeResidualScale(init_scale, max_scale, trainable_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply coordinate-channel attention and bounded residual injection."""
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = self.coord_conv(torch.cat((x_h, x_w), dim=2))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)
        coord = x * torch.sigmoid(self.coord_h(y_h)) * torch.sigmoid(self.coord_w(y_w))

        eca = x.mean(dim=(2, 3), keepdim=True)
        eca = self.eca(eca.squeeze(-1).transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        eca = x * torch.sigmoid(eca)

        enhancement = self.coord_alpha * (coord - x) + self.eca_alpha * (eca - x)
        return x + self.scale() * enhancement
