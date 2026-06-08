"""Benchmark single-image latency and FPS for a YOLO model."""

from __future__ import annotations

import argparse
import time

import torch

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_arg = str(args.device)
    if device_arg.isdigit():
        device_arg = f"cuda:{device_arg}" if torch.cuda.is_available() else "cpu"
    elif device_arg.startswith("cuda") and not torch.cuda.is_available():
        device_arg = "cpu"
    device = torch.device(device_arg)
    model = YOLO(args.weights).model.to(device).eval()
    x = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = elapsed / args.iters * 1000.0
    fps = 1000.0 / latency_ms
    params = sum(p.numel() for p in model.parameters())
    print(f"latency_ms={latency_ms:.3f}")
    print(f"fps={fps:.2f}")
    print(f"params={params}")


if __name__ == "__main__":
    main()
