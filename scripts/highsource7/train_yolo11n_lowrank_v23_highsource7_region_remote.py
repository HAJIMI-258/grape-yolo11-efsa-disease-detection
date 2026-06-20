from __future__ import annotations

import csv
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

# Import before YOLO loads a V23 checkpoint so pickle can resolve LowRankConv.
from highsource7_ap50_checkpoint import AP50CheckpointMixin
from lowrank_compression_v23 import compress_to_budget, save_lowrank_checkpoint
from train_yolo11n_gapkd_full_highsource7_region_remote import FullGapDistillCriterion

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")
SOURCE = Path(
    os.environ.get(
        "GRAPE_V23_SOURCE",
        r"D:\grape_combo\runs\yolo11n_p5internal_v22_h48_highsource7_region_img640_e150\weights\best_map50.pt",
    )
)
TEACHER = Path(
    os.environ.get(
        "GRAPE_V23_TEACHER",
        r"D:\grape_combo\runs\yolo11n_p5prune_v20_p251_highsource7_region_img640_e150\weights\best_map50.pt",
    )
)
TARGET_PARAMETERS = int(os.environ.get("GRAPE_V23_TARGET", "2100000"))
MIN_ENERGY = float(os.environ.get("GRAPE_V23_MIN_ENERGY", "0.95"))
BASELINE_AP50 = 0.96878
V22_AP50 = 0.97115


class LowRankRecoveryTrainer(AP50CheckpointMixin, DetectionTrainer):
    """Recover an output-compatible low-rank detector with the passing p251 teacher."""

    def set_model_attributes(self):
        super().set_model_attributes()
        if not TEACHER.exists():
            raise FileNotFoundError(f"V23 teacher not found: {TEACHER}")

        teacher = YOLO(str(TEACHER)).model.to(self.device)
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
            "V23 low-rank recovery KD: teacher=%s, cls=0.20, response=0.06, dfl=0.02",
            TEACHER,
        )


def _target_tag() -> str:
    return f"p{TARGET_PARAMETERS // 1000}k"


def _validate(checkpoint: Path, suffix: str) -> tuple[float, float]:
    metrics = YOLO(str(checkpoint)).val(
        data=str(DATA_CFG),
        imgsz=640,
        batch=64,
        workers=4,
        device=0,
        plots=False,
        verbose=False,
        project=str(REMOTE / "runs"),
        name=f"yolo11n_lowrank_v23_{_target_tag()}_{suffix}",
    )
    ap50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    LOGGER.info("V23 %s validation: AP50=%.5f, mAP50-95=%.5f", suffix, ap50, map5095)
    return ap50, map5095


def _write_plan(choices, before: int, after: int, checkpoint: Path) -> None:
    plan_path = checkpoint.with_suffix(".csv")
    fields = ["module", "rank", "parameters_saved", "retained_energy"]
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for choice in choices:
            writer.writerow(
                {
                    "module": choice.name,
                    "rank": choice.rank,
                    "parameters_saved": choice.parameters_saved,
                    "retained_energy": f"{choice.retained_energy:.8f}",
                }
            )
    LOGGER.info(
        "V23 plan saved to %s; params=%d -> %d (%.2f%% reduction)",
        plan_path,
        before,
        after,
        100.0 * (before - after) / before,
    )


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if TARGET_PARAMETERS >= 2_400_000 or TARGET_PARAMETERS < 1_850_000:
        raise ValueError("GRAPE_V23_TARGET must be in [1,850,000, 2,400,000)")

    source_wrapper = YOLO(str(SOURCE))
    source = source_wrapper.model.float().cpu().eval()
    before = sum(parameter.numel() for parameter in source.parameters())
    compressed, choices = compress_to_budget(source, TARGET_PARAMETERS, min_retained_energy=MIN_ENERGY)
    after = sum(parameter.numel() for parameter in compressed.parameters())
    LOGGER.info("V23 min retained energy limit: %.3f", MIN_ENERGY)

    checkpoint = REMOTE / "lowrank_models" / f"yolo11n_v23_{_target_tag()}_before_recovery.pt"
    save_lowrank_checkpoint(compressed, SOURCE, checkpoint)
    _write_plan(choices, before, after, checkpoint)
    initial_ap50, _ = _validate(checkpoint, "initial_val")
    LOGGER.info(
        "V23 initial gaps: baseline=%+.5f, V22=%+.5f",
        initial_ap50 - BASELINE_AP50,
        initial_ap50 - V22_AP50,
    )

    overrides = {
        "model": str(checkpoint),
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
        "lr0": 0.0008,
        "lrf": 0.10,
        "cos_lr": True,
        "warmup_epochs": 1.0,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        # Preserve the shallow lesion representation. Low-rank modules live in
        # deeper backbone, neck, and selected detection towers.
        "freeze": list(range(5)),
        "project": str(REMOTE / "runs"),
        "name": f"yolo11n_lowrank_v23_{_target_tag()}_highsource7_region_img640_e150",
        "exist_ok": True,
        "verbose": True,
    }

    trainer = LowRankRecoveryTrainer(overrides=overrides)
    trainer.model = compressed
    trainer.train()


if __name__ == "__main__":
    main()
