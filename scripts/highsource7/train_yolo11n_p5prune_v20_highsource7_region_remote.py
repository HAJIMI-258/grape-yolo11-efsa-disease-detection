from __future__ import annotations

import os
import sys
from pathlib import Path

REMOTE = Path(r"D:\grape_combo")
REPO = REMOTE / "grape-yolo11-efsa-disease-detection"
if REPO.exists():
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REMOTE))

from highsource7_ap50_checkpoint import AP50CheckpointMixin
from p5_depgraph_pruning import PROFILES, prune_yolo11n_p5, save_pruned_model
from train_yolo11n_gapkd_full_highsource7_region_remote import FullGapDistillCriterion

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")
AP50_SOURCE = Path(
    os.environ.get(
        "GRAPE_PRUNE_SOURCE",
        r"D:\grape_combo\runs\yolo11n_highsource7_region_ap50teacher_img640_e150\weights\best_map50.pt",
    )
)
FALLBACK_SOURCE = Path(r"D:\grape_combo\runs\yolo11n_highsource7_region_fast_img640_e150\weights\best.pt")
TEACHER_WEIGHTS = Path(r"D:\grape_combo\runs\yolo11n_highsource7_region_fast_img640_e150\weights\best.pt")
PROFILE_NAME = os.environ.get("GRAPE_P5_PROFILE", "p233").lower()


class P5PrunedRecoveryTrainer(AP50CheckpointMixin, DetectionTrainer):
    """Fine-tune a physically pruned baseline while preserving its learned P3/P4 representation."""

    def set_model_attributes(self):
        super().set_model_attributes()
        if not TEACHER_WEIGHTS.exists():
            raise FileNotFoundError(f"Recovery teacher not found: {TEACHER_WEIGHTS}")

        teacher = YOLO(str(TEACHER_WEIGHTS)).model.to(self.device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher = teacher

        # This is intentionally milder than v14. The student is already a trained
        # baseline subnetwork, so KD is a retention regularizer rather than the
        # primary learning signal.
        self.model.criterion = FullGapDistillCriterion(
            self.model,
            teacher,
            cls_weight=0.25,
            response_weight=0.08,
            dfl_weight=0.03,
            temperature=3.0,
            bg_weight=0.01,
            fg_threshold=0.30,
        )
        LOGGER.info(
            "V20 conservative recovery KD: teacher=%s, cls=0.25, response=0.08, dfl=0.03",
            TEACHER_WEIGHTS,
        )


def _source_checkpoint() -> Path:
    if AP50_SOURCE.exists():
        return AP50_SOURCE
    if FALLBACK_SOURCE.exists():
        LOGGER.warning("AP50-selected source missing; falling back to %s", FALLBACK_SOURCE)
        return FALLBACK_SOURCE
    raise FileNotFoundError(f"Neither pruning source exists: {AP50_SOURCE} or {FALLBACK_SOURCE}")


def main() -> None:
    if PROFILE_NAME not in PROFILES:
        raise KeyError(f"Unknown GRAPE_P5_PROFILE={PROFILE_NAME!r}; choose from {sorted(PROFILES)}")
    profile = PROFILES[PROFILE_NAME]
    source_path = _source_checkpoint()

    source = YOLO(str(source_path)).model.float().cpu().eval()
    before = sum(parameter.numel() for parameter in source.parameters())
    pruned = prune_yolo11n_p5(source, profile)
    after = sum(parameter.numel() for parameter in pruned.parameters())
    reduction = 100.0 * (before - after) / before
    LOGGER.info(
        "V20 %s P5-only DepGraph pruning: source=%s, params=%d -> %d (%.2f%% reduction)",
        profile.name,
        source_path,
        before,
        after,
        reduction,
    )

    audit_path = REMOTE / "pruned_models" / f"yolo11n_highsource7_{profile.name}_before_recovery.pt"
    save_pruned_model(
        pruned,
        audit_path,
        metadata={
            "profile": profile.name,
            "source": str(source_path),
            "params_before": before,
            "params_after": after,
        },
    )

    run_name = f"yolo11n_p5prune_v20_{profile.name}_highsource7_region_img640_e150"
    overrides = {
        "model": str(source_path),
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
        # Compression recovery is not a same-start architecture run. A low-LR
        # schedule prevents the retained baseline subspace from being overwritten.
        "optimizer": "SGD",
        "lr0": 0.001,
        "lrf": 0.10,
        "cos_lr": True,
        "warmup_epochs": 1.0,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        # Layers 0-6 are unpruned and contain the lesion-sensitive P2/P3/P4
        # backbone representation. Keep them fixed during recovery.
        "freeze": list(range(7)),
        "project": str(REMOTE / "runs"),
        "name": run_name,
        "exist_ok": True,
        "verbose": True,
    }

    trainer = P5PrunedRecoveryTrainer(overrides=overrides)
    trainer.model = pruned
    trainer.train()


if __name__ == "__main__":
    main()
