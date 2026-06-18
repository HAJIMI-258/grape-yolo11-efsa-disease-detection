from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch

REMOTE = Path(r"D:\grape_combo")
REPO = REMOTE / "grape-yolo11-efsa-disease-detection"
if REPO.exists():
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REMOTE))

from slim_weight_inherit import inherit_narrow_model_weights
from train_yolo11n_width20_gapkd_v12_highsource7_region_remote import (
    CLASS_KD_WEIGHTS,
    CLASS_NAMES,
    DATA_CFG,
    STUDENT_CFG,
    GapFeatureDistillCriterion,
    _select_o2m,
)

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

INIT_WEIGHTS = Path(
    os.environ.get(
        "GRAPE_SLIM_INIT_WEIGHTS",
        r"D:\grape_combo\runs\yolo11n_highsource7_region_fast_img640_e150\weights\best.pt",
    )
)
STRONG_TEACHER_WEIGHTS = Path(
    os.environ.get(
        "GRAPE_STRONG_TEACHER_WEIGHTS",
        r"D:\grape_combo\runs\yolo11s_highsource7_teacher_img640_e150\weights\best.pt",
    )
)
TEACHER_CHUNK = max(1, int(os.environ.get("GRAPE_TEACHER_CHUNK", "16")))


def _concat_prediction_parts(parts: list[dict]) -> dict:
    """Concatenate raw YOLO prediction dictionaries along the batch dimension."""
    if not parts:
        raise ValueError("No teacher prediction parts were produced")
    if len(parts) == 1:
        return parts[0]

    output: dict = {}
    first = parts[0]
    for key, value in first.items():
        if torch.is_tensor(value):
            output[key] = torch.cat([part[key] for part in parts], dim=0)
        elif isinstance(value, (list, tuple)):
            levels = []
            for level in range(len(value)):
                levels.append(torch.cat([part[key][level] for part in parts], dim=0))
            output[key] = levels
        else:
            output[key] = value
    return output


class StrongTeacherLateGapCriterion(GapFeatureDistillCriterion):
    """V14 loss with chunked YOLO11s teacher inference and late foreground feature KD."""

    def __init__(
        self,
        *args,
        feature_start_epoch: int = 80,
        feature_ramp_epochs: int = 30,
        feature_final_weight: float = 0.025,
        teacher_chunk: int = 16,
        **kwargs,
    ):
        super().__init__(*args, feature_weight=feature_final_weight, **kwargs)
        self.current_epoch = 0
        self.feature_start_epoch = feature_start_epoch
        self.feature_ramp_epochs = max(feature_ramp_epochs, 1)
        self.feature_final_weight = feature_final_weight
        self.teacher_chunk = max(1, teacher_chunk)

    def _scheduled_feature_weight(self) -> float:
        if self.current_epoch < self.feature_start_epoch:
            return 0.0
        progress = (self.current_epoch - self.feature_start_epoch + 1) / self.feature_ramp_epochs
        return self.feature_final_weight * min(1.0, max(0.0, progress))

    def _teacher_predictions(self, images: torch.Tensor) -> dict:
        parts: list[dict] = []
        head = self.teacher.model[-1]
        was_training = head.training
        head.training = True
        try:
            with torch.no_grad():
                for image_chunk in images.split(self.teacher_chunk, dim=0):
                    prediction = _select_o2m(self.teacher(image_chunk))
                    if not isinstance(prediction, dict):
                        raise TypeError(f"Expected teacher prediction dict, got {type(prediction)!r}")
                    parts.append(prediction)
        finally:
            head.training = was_training
        return _concat_prediction_parts(parts)

    def __call__(self, preds, batch):
        base_loss, loss_items = self.base(preds, batch)
        student_preds = _select_o2m(preds)
        if not torch.is_grad_enabled() or not isinstance(student_preds, dict):
            return base_loss, loss_items
        if "scores" not in student_preds or not student_preds["scores"].requires_grad:
            return base_loss, loss_items

        teacher_preds = self._teacher_predictions(batch["img"])
        head_kd = self._head_distill_loss(student_preds, teacher_preds)
        feature_kd = self._foreground_feature_loss(student_preds, teacher_preds, batch)
        feature_weight = self._scheduled_feature_weight()
        total_kd = head_kd + feature_weight * feature_kd
        return base_loss + total_kd * batch["img"].shape[0], loss_items


def _extract_map50(metrics: dict) -> float | None:
    """Read AP50 from an Ultralytics metric dictionary without matching AP50-95."""
    preferred = ("metrics/mAP50(B)", "metrics/mAP50")
    for key in preferred:
        if key in metrics:
            return float(metrics[key])
    for key, value in metrics.items():
        normalized = key.lower().replace(" ", "")
        if "map50" in normalized and "50-95" not in normalized and "5095" not in normalized:
            return float(value)
    return None


class StrongTeacherSlimInitTrainer(DetectionTrainer):
    """Attach the strong frozen teacher while keeping deployment model at 1.218M parameters."""

    def set_model_attributes(self):
        super().set_model_attributes()
        if not STRONG_TEACHER_WEIGHTS.exists():
            raise FileNotFoundError(
                f"Strong teacher not found: {STRONG_TEACHER_WEIGHTS}. "
                "Run train_yolo11s_teacher_highsource7_region_remote.py first or set "
                "GRAPE_STRONG_TEACHER_WEIGHTS."
            )
        teacher = YOLO(str(STRONG_TEACHER_WEIGHTS)).model.to(self.device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher = teacher
        self.best_map50 = float("-inf")
        self.model.criterion = StrongTeacherLateGapCriterion(
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
            teacher_chunk=TEACHER_CHUNK,
        )
        LOGGER.info(
            "GapKD v1.7 enabled: inherited 1.2M student, YOLO11s teacher=%s, "
            "chunk=%d, late P3/P4 foreground feature KD start=80 ramp=30 final=0.025",
            STRONG_TEACHER_WEIGHTS,
            TEACHER_CHUNK,
        )
        LOGGER.info("Class KD weights: %s", dict(zip(CLASS_NAMES, CLASS_KD_WEIGHTS)))

    def run_callbacks(self, event: str):
        if event == "on_train_epoch_start" and hasattr(self.model, "criterion"):
            criterion = self.model.criterion
            if hasattr(criterion, "current_epoch"):
                criterion.current_epoch = int(self.epoch) + 1
        return super().run_callbacks(event)

    def save_model(self):
        saved = super().save_model()
        map50 = _extract_map50(self.metrics)
        if saved and map50 is not None and map50 > self.best_map50:
            self.best_map50 = map50
            destination = self.wdir / "best_map50.pt"
            shutil.copy2(self.last, destination)
            LOGGER.info("Saved AP50-best checkpoint: %.6f -> %s", map50, destination)
        return saved


def main() -> None:
    if not INIT_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Slim initialization checkpoint not found: {INIT_WEIGHTS}. "
            "Set GRAPE_SLIM_INIT_WEIGHTS to the trained YOLO11n baseline checkpoint."
        )

    student = YOLO(str(STUDENT_CFG))
    source = YOLO(str(INIT_WEIGHTS)).model
    stats = inherit_narrow_model_weights(
        student.model,
        source,
        target_num_classes=len(CLASS_NAMES),
    )
    LOGGER.info(
        "Width-aware inheritance from %s: exact=%d sliced=%d skipped=%d coverage=%.2f%%",
        INIT_WEIGHTS,
        stats.exact_tensors,
        stats.sliced_tensors,
        stats.skipped_tensors,
        100.0 * stats.element_coverage,
    )
    del source

    student.train(
        trainer=StrongTeacherSlimInitTrainer,
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
        name="yolo11n_width20_gapkd_v17_strongteacher_sliminit_highsource7_region_img640_e150",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
