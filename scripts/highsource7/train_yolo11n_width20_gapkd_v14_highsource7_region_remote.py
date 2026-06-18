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


class LateFeatureGapCriterion(GapFeatureDistillCriterion):
    """Keep v1 head KD early, then ramp in low-weight P3/P4 foreground feature KD."""

    def __init__(
        self,
        *args,
        feature_start_epoch: int = 80,
        feature_ramp_epochs: int = 30,
        feature_final_weight: float = 0.025,
        **kwargs,
    ):
        super().__init__(*args, feature_weight=feature_final_weight, **kwargs)
        self.current_epoch = 0
        self.feature_start_epoch = feature_start_epoch
        self.feature_ramp_epochs = max(feature_ramp_epochs, 1)
        self.feature_final_weight = feature_final_weight

    def _scheduled_feature_weight(self) -> float:
        if self.current_epoch < self.feature_start_epoch:
            return 0.0
        progress = (self.current_epoch - self.feature_start_epoch + 1) / self.feature_ramp_epochs
        return self.feature_final_weight * min(1.0, max(0.0, progress))

    def __call__(self, preds, batch):
        old_weight = self.feature_weight
        self.feature_weight = self._scheduled_feature_weight()
        try:
            return super().__call__(preds, batch)
        finally:
            self.feature_weight = old_weight


class GapFeatureDistillV14Trainer(DetectionTrainer):
    """v1 head KD with late-ramped light P3/P4 feature KD."""

    def set_model_attributes(self):
        super().set_model_attributes()
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
            "GapKD v1.4 enabled: teacher=%s, v1 head KD plus late-ramped foreground P3/P4 "
            "feature KD, start_epoch=80, ramp=30, final_feature_weight=0.025",
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
        trainer=GapFeatureDistillV14Trainer,
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
        name="yolo11n_width20_gapkd_v14_highsource7_region_img640_e150",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
