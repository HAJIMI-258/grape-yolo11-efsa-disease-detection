"""Train Grape-EFSA-YOLO11."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data/grape.yaml")
    parser.add_argument("--model", default="configs/models/yolo11n_efsa_grape.yaml")
    parser.add_argument("--pretrained", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/grape")
    parser.add_argument("--name", default="yolo11n_efsa_768_seed42")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    if args.pretrained:
        try:
            model.load(args.pretrained)
        except Exception as exc:  # noqa: BLE001
            print(f"Partial pretrained loading failed, continuing from YAML init: {exc}")
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        seed=args.seed,
        device=args.device,
        project=args.project,
        name=args.name,
        amp=True,
        cos_lr=True,
        close_mosaic=20,
    )


if __name__ == "__main__":
    main()
