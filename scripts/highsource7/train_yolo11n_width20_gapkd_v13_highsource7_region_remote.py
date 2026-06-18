from __future__ import annotations

import sys
from pathlib import Path


REMOTE = Path(r"D:\grape_combo")
REPO = REMOTE / "grape-yolo11-efsa-disease-detection"
if REPO.exists():
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REMOTE))

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

from train_yolo11n_width20_gapkd_v12_highsource7_region_remote import (
    CLASS_KD_WEIGHTS,
    CLASS_NAMES,
    DATA_CFG,
    STUDENT_CFG,
    TEACHER_WEIGHTS,
    GapFeatureDistillCriterion,
)


class GapFeatureDistillV13Trainer(DetectionTrainer):
    """v1 head KD with a very light P3/P4 feature term."""

    def set_model_attributes(self):
        super().set_model_attributes()
        teacher = YOLO(str(TEACHER_WEIGHTS)).model.to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher
        self.model.criterion = GapFeatureDistillCriterion(
            self.model,
            teacher,
            cls_weight=0.70,
            response_weight=0.18,
            dfl_weight=0.06,
            feature_weight=0.02,
            temperature=3.0,
            bg_weight=0.03,
            fg_threshold=0.25,
            gt_expand=1.40,
        )
        LOGGER.info(
            "GapKD v1.3 enabled: teacher=%s, v1 head KD + light foreground P3/P4 feature KD, "
            "feature_weight=0.02, gt_expand=1.40, no hard negatives",
            TEACHER_WEIGHTS,
        )
        LOGGER.info("Class KD weights: %s", dict(zip(CLASS_NAMES, CLASS_KD_WEIGHTS)))


def main() -> None:
    model = YOLO(str(STUDENT_CFG))
    model.train(
        trainer=GapFeatureDistillV13Trainer,
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
        name="yolo11n_width20_gapkd_v13_highsource7_region_img640_e150",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
