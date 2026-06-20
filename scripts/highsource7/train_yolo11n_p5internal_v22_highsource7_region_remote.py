from __future__ import annotations

import os
import sys
from pathlib import Path

REMOTE = Path(r"D:\grape_combo")
REPO = REMOTE / "grape-yolo11-efsa-disease-detection"
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
if REPO.exists():
    sys.path.insert(1, str(REPO))
sys.path.insert(2, str(REMOTE))

from highsource7_ap50_checkpoint import AP50CheckpointMixin
from p5_internal_pruning_v22 import PROFILES, prune_yolo11n_p5_internal, save_pruned_ultralytics_checkpoint
from train_yolo11n_gapkd_full_highsource7_region_remote import FullGapDistillCriterion

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")
SOURCE = Path(
    os.environ.get(
        "GRAPE_V22_SOURCE",
        r"D:\grape_combo\runs\yolo11n_p5prune_v20_p251_highsource7_region_img640_e150\weights\best_map50.pt",
    )
)
TEACHER_WEIGHTS = Path(
    os.environ.get(
        "GRAPE_V22_TEACHER",
        r"D:\grape_combo\runs\yolo11n_p5prune_v20_p251_highsource7_region_img640_e150\weights\best_map50.pt",
    )
)
PROFILE_NAME = os.environ.get("GRAPE_V22_PROFILE", "h56").lower()


class P5InternalRecoveryTrainer(AP50CheckpointMixin, DetectionTrainer):
    """Recover a three-scale p251 model after P5-internal structured pruning."""

    def set_model_attributes(self):
        super().set_model_attributes()
        if not TEACHER_WEIGHTS.exists():
            raise FileNotFoundError(f"Recovery teacher not found: {TEACHER_WEIGHTS}")

        teacher = YOLO(str(TEACHER_WEIGHTS)).model.to(self.device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher = teacher

        self.model.criterion = FullGapDistillCriterion(
            self.model,
            teacher,
            cls_weight=0.20,
            response_weight=0.06,
            dfl_weight=0.02,
            temperature=3.0,
            bg_weight=0.01,
            fg_threshold=0.30,
        )
        LOGGER.info(
            "V22 P5-internal recovery KD: teacher=%s, cls=0.20, response=0.06, dfl=0.02",
            TEACHER_WEIGHTS,
        )


def _validate_before_recovery(pruned, source_path: Path, profile_name: str) -> tuple[Path, float, float]:
    checkpoint_path = REMOTE / "pruned_models" / f"yolo11n_highsource7_v22_{profile_name}_before_recovery.pt"
    save_pruned_ultralytics_checkpoint(pruned, source_path, checkpoint_path)
    metrics = YOLO(str(checkpoint_path)).val(
        data=str(DATA_CFG),
        imgsz=640,
        batch=64,
        workers=4,
        device=0,
        plots=False,
        verbose=False,
        project=str(REMOTE / "runs"),
        name=f"yolo11n_p5internal_v22_{profile_name}_initial_val",
    )
    ap50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    LOGGER.info("V22 %s immediate validation: AP50=%.5f, mAP50-95=%.5f", profile_name, ap50, map5095)
    return checkpoint_path, ap50, map5095


def main() -> None:
    if PROFILE_NAME not in PROFILES:
        raise KeyError(f"Unknown GRAPE_V22_PROFILE={PROFILE_NAME!r}; choose from {sorted(PROFILES)}")
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    profile = PROFILES[PROFILE_NAME]
    source = YOLO(str(SOURCE)).model.float().cpu().eval()
    before = sum(parameter.numel() for parameter in source.parameters())
    pruned = prune_yolo11n_p5_internal(source, profile)
    after = sum(parameter.numel() for parameter in pruned.parameters())
    LOGGER.info(
        "V22 %s P5-internal pruning: source=%s, params=%d -> %d (%.2f%% reduction)",
        profile.name,
        SOURCE,
        before,
        after,
        100.0 * (before - after) / before,
    )
    _validate_before_recovery(pruned, SOURCE, profile.name)

    run_name = f"yolo11n_p5internal_v22_{profile.name}_highsource7_region_img640_e150"
    overrides = {
        "model": str(SOURCE),
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
        # Keep the lesion-sensitive shallow and mid-level representation fixed.
        "freeze": list(range(7)),
        "project": str(REMOTE / "runs"),
        "name": run_name,
        "exist_ok": True,
        "verbose": True,
    }

    trainer = P5InternalRecoveryTrainer(overrides=overrides)
    trainer.model = pruned
    trainer.train()


if __name__ == "__main__":
    main()
