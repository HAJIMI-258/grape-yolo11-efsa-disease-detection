from __future__ import annotations

import sys
from pathlib import Path

REMOTE = Path(r"D:\grape_combo")
REPO = REMOTE / "grape-yolo11-efsa-disease-detection"
if REPO.exists():
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REMOTE))

from highsource7_ap50_checkpoint import AP50CheckpointMixin
from train_yolo11n_width20_gapkd_v12_highsource7_region_remote import (
    CLASS_KD_WEIGHTS,
    CLASS_NAMES,
    DATA_CFG,
    STUDENT_CFG,
)
from train_yolo11n_width20_gapkd_v14_highsource7_region_remote import LateFeatureGapCriterion

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

TEACHER_WEIGHTS = Path(r"D:\grape_combo\runs\yolo11n_highsource7_region_ap50teacher_img640_e150\weights\best_map50.pt")


class GapFeatureDistillV16Trainer(AP50CheckpointMixin, DetectionTrainer):
    """V14 schedule with an AP50-selected baseline teacher and AP50-selected student checkpoint."""

    def set_model_attributes(self):
        super().set_model_attributes()
        if not TEACHER_WEIGHTS.exists():
            raise FileNotFoundError(
                f"AP50 teacher checkpoint not found: {TEACHER_WEIGHTS}. "
                "Run train_yolo11n_highsource7_region_ap50_teacher_remote.py first."
            )

        teacher = YOLO(str(TEACHER_WEIGHTS)).model.to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher
        self.model.criterion = LateFeatureGapCriterion(
            self.model,
            teacher,
            cls_weight=0.70,
            response_weight=0.18,
            dfl_weight=0.06,
            temperature=3.0,
            bg_weight=0.03,
            fg_threshold=0.25,
            gt_expand=1.40,
            feature_start_epoch=80,
            feature_ramp_epochs=30,
            feature_final_weight=0.025,
        )
        LOGGER.info(
            "GapKD v1.6 enabled: teacher=%s, v14 late-ramped foreground P3/P4 feature KD, "
            "student also saves best_map50.pt",
            TEACHER_WEIGHTS,
        )
        LOGGER.info("Class KD weights: %s", dict(zip(CLASS_NAMES, CLASS_KD_WEIGHTS)))

    def run_callbacks(self, event: str):
        if event == "on_train_epoch_start" and hasattr(self.model, "criterion"):
            criterion = self.model.criterion
            if hasattr(criterion, "current_epoch"):
                criterion.current_epoch = int(self.epoch) + 1
        return super().run_callbacks(event)


def main() -> None:
    model = YOLO(str(STUDENT_CFG))
    model.train(
        trainer=GapFeatureDistillV16Trainer,
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
        name="yolo11n_width20_gapkd_v16_ap50teacher_highsource7_region_img640_e150",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
