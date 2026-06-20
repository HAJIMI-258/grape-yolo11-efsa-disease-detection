from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REMOTE = Path(r"D:\grape_combo")
REPO = REMOTE / "grape-yolo11-efsa-disease-detection"
if REPO.exists():
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REMOTE))

from highsource7_ap50_checkpoint import AP50CheckpointMixin
from p3p4_compaction import compact_p251_to_p3p4, save_compact_checkpoint
from train_yolo11n_width20_gapkd_v12_highsource7_region_remote import GapFeatureDistillCriterion, _select_o2m

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")
P251_SOURCE = Path(
    os.environ.get(
        "GRAPE_P251_SOURCE",
        r"D:\grape_combo\runs\yolo11n_p5prune_v20_p251_highsource7_region_img640_e150\weights\best_map50.pt",
    )
)
BASELINE_AP50 = 0.96878


class TwoScaleRetentionCriterion(GapFeatureDistillCriterion):
    """Conservative P3/P4 retention KD from the three-scale p251 teacher."""

    def __init__(self, student, teacher):
        super().__init__(
            student,
            teacher,
            cls_weight=0.25,
            response_weight=0.08,
            dfl_weight=0.03,
            feature_weight=0.01,
            temperature=3.0,
            bg_weight=0.01,
            fg_threshold=0.30,
            gt_expand=1.25,
        )
        # The p251 teacher is already strong on all seven classes. Do not repeat
        # the aggressive weak-class weighting that destabilized prior variants.
        self.class_weights = torch.ones(student.nc, dtype=torch.float32)

    @staticmethod
    def _trim_teacher(teacher_preds: dict, student_preds: dict) -> dict:
        """Keep the teacher's P3/P4 anchors and features in the same order as the student."""
        anchors = student_preds["scores"].shape[-1]
        trimmed = dict(teacher_preds)
        trimmed["scores"] = teacher_preds["scores"][..., :anchors]
        trimmed["boxes"] = teacher_preds["boxes"][..., :anchors]
        if teacher_preds.get("feats"):
            trimmed["feats"] = list(teacher_preds["feats"][: len(student_preds.get("feats", []))])
        return trimmed

    def __call__(self, preds, batch):
        base_loss, loss_items = self.base(preds, batch)
        student_preds = _select_o2m(preds)
        if not torch.is_grad_enabled() or not isinstance(student_preds, dict):
            return base_loss, loss_items
        if "scores" not in student_preds or not student_preds["scores"].requires_grad:
            return base_loss, loss_items

        with torch.no_grad():
            head = self.teacher.model[-1]
            was_training = head.training
            head.training = True
            teacher_preds = _select_o2m(self.teacher(batch["img"]))
            head.training = was_training
        if not isinstance(teacher_preds, dict):
            return base_loss, loss_items

        teacher_preds = self._trim_teacher(teacher_preds, student_preds)
        head_kd = self._head_distill_loss(student_preds, teacher_preds)
        feature_kd = self._foreground_feature_loss(student_preds, teacher_preds, batch)
        total_kd = head_kd + self.feature_weight * feature_kd
        return base_loss + total_kd * batch["img"].shape[0], loss_items


class P3P4CompactRecoveryTrainer(AP50CheckpointMixin, DetectionTrainer):
    """Recover the compact two-scale detector while freezing its retained backbone."""

    def set_model_attributes(self):
        super().set_model_attributes()
        if not P251_SOURCE.exists():
            raise FileNotFoundError(f"p251 teacher/source not found: {P251_SOURCE}")

        teacher = YOLO(str(P251_SOURCE)).model.to(self.device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher = teacher
        self.model.criterion = TwoScaleRetentionCriterion(self.model, teacher)
        LOGGER.info(
            "V21 P3/P4 retention KD: teacher=%s, cls=0.25, response=0.08, dfl=0.03, feature=0.01",
            P251_SOURCE,
        )


def _validate_compact(checkpoint: Path) -> tuple[float, float]:
    metrics = YOLO(str(checkpoint)).val(
        data=str(DATA_CFG),
        imgsz=640,
        batch=64,
        workers=4,
        device=0,
        plots=False,
        verbose=False,
        project=str(REMOTE / "runs"),
        name="yolo11n_p3p4compact_v21_initial_val",
    )
    ap50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    LOGGER.info("V21 immediate compact validation: AP50=%.5f, mAP50-95=%.5f", ap50, map5095)
    return ap50, map5095


def main() -> None:
    if not P251_SOURCE.exists():
        raise FileNotFoundError(P251_SOURCE)

    source = YOLO(str(P251_SOURCE)).model.float().cpu().eval()
    before = sum(parameter.numel() for parameter in source.parameters())
    compact = compact_p251_to_p3p4(source)
    after = sum(parameter.numel() for parameter in compact.parameters())
    reduction = 100.0 * (before - after) / before
    LOGGER.info("V21 P3/P4 compaction: params=%d -> %d (%.2f%% reduction)", before, after, reduction)

    compact_checkpoint = REMOTE / "compact_models" / "yolo11n_p251_p3p4compact_v21_before_recovery.pt"
    save_compact_checkpoint(compact, P251_SOURCE, compact_checkpoint)
    initial_ap50, _ = _validate_compact(compact_checkpoint)
    LOGGER.info(
        "V21 initial hard-requirement gap: AP50 %.5f versus baseline %.5f (%+.5f)",
        initial_ap50,
        BASELINE_AP50,
        initial_ap50 - BASELINE_AP50,
    )

    overrides = {
        "model": str(compact_checkpoint),
        "data": str(DATA_CFG),
        "imgsz": 640,
        "epochs": 150,
        "batch": 64,
        "workers": 4,
        "device": 0,
        "pretrained": False,
        "patience": 0,
        "amp": True,
        "cache": False,
        "seed": 42,
        "box": 8.0,
        "cls": 0.65,
        "dfl": 1.7,
        "mosaic": 0.45,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 20,
        "optimizer": "SGD",
        "lr0": 0.001,
        "lrf": 0.10,
        "cos_lr": True,
        "warmup_epochs": 1.0,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        # Retain the complete trained backbone and only adapt the P3/P4 neck/head
        # to absorb objects previously emitted by the discarded P5 tower.
        "freeze": list(range(11)),
        "project": str(REMOTE / "runs"),
        "name": "yolo11n_p3p4compact_v21_highsource7_region_img640_e150",
        "exist_ok": True,
        "verbose": True,
    }

    trainer = P3P4CompactRecoveryTrainer(overrides=overrides)
    trainer.model = compact
    trainer.train()


if __name__ == "__main__":
    main()
