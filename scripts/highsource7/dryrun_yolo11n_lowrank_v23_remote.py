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

from lowrank_compression_v23 import compress_to_budget, save_lowrank_checkpoint

from ultralytics import YOLO

DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")
SOURCE = Path(
    os.environ.get(
        "GRAPE_V23_SOURCE",
        r"D:\grape_combo\runs\yolo11n_p5internal_v22_h48_highsource7_region_img640_e150\weights\best_map50.pt",
    )
)
TARGET = int(os.environ.get("GRAPE_V23_TARGET", "2100000"))
MIN_ENERGY = float(os.environ.get("GRAPE_V23_MIN_ENERGY", "0.95"))
DEEP_MIN_ENERGY = os.environ.get("GRAPE_V23_DEEP_MIN_ENERGY")
DEEP_MIN_ENERGY_VALUE = float(DEEP_MIN_ENERGY) if DEEP_MIN_ENERGY else None
RANK_STEP = int(os.environ.get("GRAPE_V23_RANK_STEP", "8"))
OUTPUT = REMOTE / "lowrank_models" / f"yolo11n_v23_p{TARGET // 1000}k_dryrun.pt"


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    wrapper = YOLO(str(SOURCE))
    model = wrapper.model.float().cpu().eval()
    before = sum(parameter.numel() for parameter in model.parameters())
    compressed, choices = compress_to_budget(
        model,
        TARGET,
        min_retained_energy=MIN_ENERGY,
        deep_min_retained_energy=DEEP_MIN_ENERGY_VALUE,
        rank_step=RANK_STEP,
    )
    after = sum(parameter.numel() for parameter in compressed.parameters())
    save_lowrank_checkpoint(compressed, SOURCE, OUTPUT)

    plan_csv = OUTPUT.with_suffix(".csv")
    with plan_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["module", "rank", "parameters_saved", "retained_energy"])
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

    loaded = YOLO(str(OUTPUT))
    summary = loaded.info(imgsz=640, verbose=False)
    metrics = loaded.val(
        data=str(DATA_CFG),
        imgsz=640,
        batch=64,
        workers=4,
        device=0,
        plots=False,
        verbose=False,
        project=str(REMOTE / "runs"),
        name=f"yolo11n_lowrank_v23_p{TARGET // 1000}k_dryrun_val",
    )

    print("source", SOURCE)
    print("target_parameters", TARGET)
    print("min_retained_energy_limit", MIN_ENERGY)
    print("deep_min_retained_energy_limit", DEEP_MIN_ENERGY_VALUE)
    print("rank_step", RANK_STEP)
    print("params_before", before)
    print("params_after", after)
    print("reduction_percent", 100.0 * (before - after) / before)
    print("factorized_modules", len(choices))
    print("minimum_retained_energy", min(choice.retained_energy for choice in choices))
    print("model_info", summary)
    print("precision", float(metrics.box.mp))
    print("recall", float(metrics.box.mr))
    print("ap50", float(metrics.box.map50))
    print("map50_95", float(metrics.box.map))
    print("checkpoint", OUTPUT)
    print("plan", plan_csv)


if __name__ == "__main__":
    main()
