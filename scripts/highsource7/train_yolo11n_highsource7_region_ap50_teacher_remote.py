from __future__ import annotations

import sys
from pathlib import Path

REMOTE = Path(r"D:\grape_combo")
REPO = REMOTE / "grape-yolo11-efsa-disease-detection"
if REPO.exists():
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REMOTE))

from highsource7_ap50_checkpoint import AP50CheckpointMixin

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer


class AP50BaselineTrainer(AP50CheckpointMixin, DetectionTrainer):
    """Clean YOLO11n baseline that also saves weights/best_map50.pt."""


def main() -> None:
    model = YOLO(r"D:\grape_mypfe6\yolo11n.pt")
    model.train(
        trainer=AP50BaselineTrainer,
        data=r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml",
        imgsz=640,
        epochs=150,
        batch=64,
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
        name="yolo11n_highsource7_region_ap50teacher_img640_e150",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
