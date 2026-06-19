from __future__ import annotations

import sys
from pathlib import Path

REMOTE = Path(r"D:\grape_combo")
REPO = REMOTE / "grape-yolo11-efsa-disease-detection"
if REPO.exists():
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REMOTE))

from finegrained_detect import replace_detect_with_finegrained
from highsource7_ap50_checkpoint import AP50CheckpointMixin
from train_yolo11n_width20_gapkd_v12_highsource7_region_remote import DATA_CFG, STUDENT_CFG
from train_yolo11n_width20_gapkd_v14_highsource7_region_remote import GapFeatureDistillV14Trainer

from ultralytics import YOLO
from ultralytics.utils import LOGGER

CLS_CHANNELS = 96


class FineGrainedHeadV19Trainer(AP50CheckpointMixin, GapFeatureDistillV14Trainer):
    """V14 training with extra capacity only in the classification towers."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model = replace_detect_with_finegrained(model, cls_channels=CLS_CHANNELS)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        LOGGER.info(
            "V19A fine-grained classification head enabled: cls_channels=%d, params=%d",
            CLS_CHANNELS,
            parameters,
        )
        if not 1_200_000 <= parameters <= 1_350_000:
            raise RuntimeError(f"Unexpected v19A parameter count: {parameters:,}")
        return model


def main() -> None:
    model = YOLO(str(STUDENT_CFG))
    model.train(
        trainer=FineGrainedHeadV19Trainer,
        data=str(DATA_CFG),
        imgsz=640,
        epochs=150,
        batch=64,
        workers=4,
        device=0,
        pretrained=False,
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
        name="yolo11n_width20_gapkd_v19a_fghead_highsource7_region_img640_e150",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
