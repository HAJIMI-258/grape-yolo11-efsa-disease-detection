from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO

DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")
TEACHER_INIT = Path(os.environ.get("GRAPE_YOLO11S_WEIGHTS", r"D:\grape_mypfe6\yolo11s.pt"))


def main() -> None:
    if not TEACHER_INIT.exists():
        raise FileNotFoundError(
            f"Strong-teacher checkpoint not found: {TEACHER_INIT}. "
            "Set GRAPE_YOLO11S_WEIGHTS to a local yolo11s.pt path."
        )

    model = YOLO(str(TEACHER_INIT))
    model.train(
        data=str(DATA_CFG),
        imgsz=640,
        epochs=150,
        batch=32,
        nbs=64,
        workers=4,
        device=0,
        pretrained=True,
        patience=0,
        amp=True,
        cache=False,
        seed=42,
        box=8.0,
        cls=0.65,
        dfl=1.7,
        mosaic=0.45,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=20,
        project=r"D:\grape_combo\runs",
        name="yolo11s_highsource7_teacher_img640_e150",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
