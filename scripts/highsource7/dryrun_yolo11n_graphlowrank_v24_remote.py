from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import torch

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

from ultralytics import YOLO

DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")
SOURCE = Path(
    os.environ.get(
        "GRAPE_V24_SOURCE",
        r"D:\grape_combo\lowrank_models\yolo11n_v23_p1990k_dryrun.pt",
    )
)
TARGET = int(os.environ.get("GRAPE_V24_TARGET", "1850000"))
PRUNE_STEP = int(os.environ.get("GRAPE_V24_PRUNE_STEP", "4"))
MIN_RANK = int(os.environ.get("GRAPE_V24_MIN_RANK", "8"))
MIN_IMPORTANCE = float(os.environ.get("GRAPE_V24_MIN_IMPORTANCE", "0.92"))
TAG = os.environ.get("GRAPE_V24_TAG", "from_source")
OUTPUT = REMOTE / "graph_lowrank_models" / f"yolo11n_v24_{TAG}_p{TARGET // 1000}k_dryrun.pt"


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
    checkpoint = save_graph_lowrank_checkpoint(pruned, SOURCE, OUTPUT)
    plan_path = OUTPUT.with_suffix(".csv")
    _write_plan(plan_path, choices)

    with torch.no_grad():
        raw_output = pruned(torch.zeros(1, 3, 640, 640))
    raw = raw_output[1] if isinstance(raw_output, tuple) else raw_output

    summary = YOLO(str(checkpoint)).info(imgsz=640, verbose=False)
    metrics = YOLO(str(checkpoint)).val(
        data=str(DATA_CFG),
        imgsz=640,
        batch=64,
        workers=4,
        device=0,
        plots=False,
        verbose=False,
        project=str(REMOTE / "runs"),
        name=f"yolo11n_graphlowrank_v24_{TAG}_p{TARGET // 1000}k_dryrun_val",
        exist_ok=True,
    )

    print("source", SOURCE)
    print("tag", TAG)
    print("target_parameters", TARGET)
    print("prune_step", PRUNE_STEP)
    print("min_rank", MIN_RANK)
    print("min_retained_importance", MIN_IMPORTANCE)
    print("params_before", before)
    print("params_after", after)
    print("reduction_percent", 100.0 * (before - after) / before)
    print("steps", len(choices))
    print("score_shape", tuple(raw["scores"].shape))
    print("box_shape", tuple(raw["boxes"].shape))
    print("model_info", summary)
    print("precision", float(metrics.box.mp))
    print("recall", float(metrics.box.mr))
    print("ap50", float(metrics.box.map50))
    print("map50_95", float(metrics.box.map))
    print("checkpoint", checkpoint)
    print("plan", plan_path)


if __name__ == "__main__":
    main()
