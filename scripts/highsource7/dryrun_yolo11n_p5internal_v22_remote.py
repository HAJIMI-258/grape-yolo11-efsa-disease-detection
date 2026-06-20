from __future__ import annotations

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

from p5_internal_pruning_v22 import PROFILES, prune_yolo11n_p5_internal, save_pruned_ultralytics_checkpoint

from ultralytics import YOLO

DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")
SOURCE = Path(
    os.environ.get(
        "GRAPE_V22_SOURCE",
        r"D:\grape_combo\runs\yolo11n_p5prune_v20_p251_highsource7_region_img640_e150\weights\best_map50.pt",
    )
)
PROFILE_NAME = os.environ.get("GRAPE_V22_PROFILE", "h56").lower()


def main() -> None:
    if PROFILE_NAME not in PROFILES:
        raise KeyError(f"Unknown profile {PROFILE_NAME!r}; choose from {sorted(PROFILES)}")
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    source = YOLO(str(SOURCE)).model.float().cpu().eval()
    before = sum(parameter.numel() for parameter in source.parameters())
    pruned = prune_yolo11n_p5_internal(source, PROFILES[PROFILE_NAME])
    after = sum(parameter.numel() for parameter in pruned.parameters())

    with torch.no_grad():
        output = pruned(torch.zeros(1, 3, 640, 640))
    raw = output[1] if isinstance(output, tuple) else output

    checkpoint = REMOTE / "pruned_models" / f"yolo11n_highsource7_v22_{PROFILE_NAME}_before_recovery.pt"
    save_pruned_ultralytics_checkpoint(pruned, SOURCE, checkpoint)
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
        name=f"yolo11n_p5internal_v22_{PROFILE_NAME}_dryrun_val",
    )

    print("profile", PROFILE_NAME)
    print("source", SOURCE)
    print("params_before", before)
    print("params_after", after)
    print("reduction_percent", 100.0 * (before - after) / before)
    print("score_shape", tuple(raw["scores"].shape))
    print("box_shape", tuple(raw["boxes"].shape))
    print("model_info", summary)
    print("precision", float(metrics.box.mp))
    print("recall", float(metrics.box.mr))
    print("ap50", float(metrics.box.map50))
    print("map50_95", float(metrics.box.map))
    print("checkpoint", checkpoint)


if __name__ == "__main__":
    main()
