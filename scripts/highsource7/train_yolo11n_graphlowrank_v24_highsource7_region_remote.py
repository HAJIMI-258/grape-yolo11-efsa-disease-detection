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

# Import before YOLO loads a V23/V24 checkpoint so pickle can resolve LowRankConv.
import lowrank_compression_v23  # noqa: F401
from graph_lowrank_pruning_v24 import graph_prune_lowrank_ranks, save_graph_lowrank_checkpoint
from highsource7_ap50_checkpoint import AP50CheckpointMixin
from train_yolo11n_gapkd_full_highsource7_region_remote import FullGapDistillCriterion

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")
SOURCE = Path(
    os.environ.get(
        "GRAPE_V24_SOURCE",
        r"D:\grape_combo\lowrank_models\yolo11n_v23_p1990k_dryrun.pt",
    )
)
TEACHER = Path(
    os.environ.get(
        "GRAPE_V24_TEACHER",
        r"D:\grape_combo\runs\yolo11n_lowrank_v23_p2200k_highsource7_region_img640_e150\weights\best_map50.pt",
    )
)
TARGET = int(os.environ.get("GRAPE_V24_TARGET", "1850000"))
PRUNE_STEP = int(os.environ.get("GRAPE_V24_PRUNE_STEP", "4"))
MIN_RANK = int(os.environ.get("GRAPE_V24_MIN_RANK", "8"))
MIN_IMPORTANCE = float(os.environ.get("GRAPE_V24_MIN_IMPORTANCE", "0.92"))
EPOCHS = int(os.environ.get("GRAPE_V24_EPOCHS", "150"))
TAG = os.environ.get("GRAPE_V24_TAG", "from_source")


class V24GraphLowRankRecoveryTrainer(AP50CheckpointMixin, DetectionTrainer):
    """Conservative recovery for a V24 graph-pruned low-rank detector."""

    def set_model_attributes(self):
        super().set_model_attributes()
        if not TEACHER.exists():
            raise FileNotFoundError(f"V24 teacher not found: {TEACHER}")
        teacher = YOLO(str(TEACHER)).model.to(self.device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher = teacher
        self.model.criterion = FullGapDistillCriterion(
            self.model,
            teacher,
            cls_weight=0.08,
            response_weight=0.02,
            dfl_weight=0.01,
            temperature=3.0,
            bg_weight=0.01,
            fg_threshold=0.30,
        )
        LOGGER.info(
            "V24 recovery KD: teacher=%s, cls=0.08, response=0.02, dfl=0.01",
            TEACHER,
        )


def _write_plan(path: Path, choices) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "old_rank",
                "new_rank",
                "parameters_saved",
                "retained_importance",
                "pruned_indices",
            ],
        )
        writer.writeheader()
        for choice in choices:
            writer.writerow(
                {
                    "name": choice.name,
                    "old_rank": choice.old_rank,
                    "new_rank": choice.new_rank,
                    "parameters_saved": choice.parameters_saved,
                    "retained_importance": f"{choice.retained_importance:.8f}",
                    "pruned_indices": " ".join(str(index) for index in choice.pruned_indices),
                }
            )


def _validate_before_recovery(pruned, source_path: Path, target: int) -> tuple[Path, float, float]:
    checkpoint = REMOTE / "graph_lowrank_models" / f"yolo11n_v24_{TAG}_p{target // 1000}k_before_recovery.pt"
    save_graph_lowrank_checkpoint(pruned, source_path, checkpoint)
    metrics = YOLO(str(checkpoint)).val(
        data=str(DATA_CFG),
        imgsz=640,
        batch=64,
        workers=4,
        device=0,
        plots=False,
        verbose=False,
        project=str(REMOTE / "runs"),
        name=f"yolo11n_graphlowrank_v24_{TAG}_p{target // 1000}k_initial_val",
        exist_ok=True,
    )
    ap50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    LOGGER.info("V24 p%d initial validation: AP50=%.5f, mAP50-95=%.5f", target // 1000, ap50, map5095)
    return checkpoint, ap50, map5095


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    source = YOLO(str(SOURCE)).model.float().cpu().eval()
    before = sum(parameter.numel() for parameter in source.parameters())
    pruned, choices = graph_prune_lowrank_ranks(
        source,
        TARGET,
        prune_step=PRUNE_STEP,
        min_rank=MIN_RANK,
        min_retained_importance=MIN_IMPORTANCE,
    )
    after = sum(parameter.numel() for parameter in pruned.parameters())
    LOGGER.info(
        "V24 graph-lowrank pruning: source=%s, params=%d -> %d (%.2f%% reduction), steps=%d",
        SOURCE,
        before,
        after,
        100.0 * (before - after) / before,
        len(choices),
    )
    _write_plan(REMOTE / "graph_lowrank_models" / f"yolo11n_v24_{TAG}_p{TARGET // 1000}k_before_recovery.csv", choices)
    _validate_before_recovery(pruned, SOURCE, TARGET)

    run_name = f"yolo11n_graphlowrank_v24_{TAG}_p{TARGET // 1000}k_highsource7_region_img640_e{EPOCHS}"
    overrides = {
        "model": str(SOURCE),
        "data": str(DATA_CFG),
        "imgsz": 640,
        "epochs": EPOCHS,
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
        "lr0": 0.0002,
        "lrf": 0.10,
        "cos_lr": True,
        "warmup_epochs": 1.0,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "freeze": list(range(7)),
        "project": str(REMOTE / "runs"),
        "name": run_name,
        "exist_ok": True,
        "verbose": True,
    }

    trainer = V24GraphLowRankRecoveryTrainer(overrides=overrides)
    trainer.model = pruned
    trainer.train()


if __name__ == "__main__":
    main()
