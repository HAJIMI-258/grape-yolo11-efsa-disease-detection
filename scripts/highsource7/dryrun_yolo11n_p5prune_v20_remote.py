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

from p5_depgraph_pruning import PROFILES, prune_yolo11n_p5

from ultralytics import YOLO

SOURCE = Path(
    os.environ.get(
        "GRAPE_PRUNE_SOURCE",
        r"D:\grape_combo\runs\yolo11n_highsource7_region_ap50teacher_img640_e150\weights\best_map50.pt",
    )
)
PROFILE_NAME = os.environ.get("GRAPE_P5_PROFILE", "p233").lower()


def main() -> None:
    if PROFILE_NAME not in PROFILES:
        raise KeyError(f"Unknown profile {PROFILE_NAME!r}; choose from {sorted(PROFILES)}")
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    model = YOLO(str(SOURCE)).model.float().cpu().eval()
    before = sum(parameter.numel() for parameter in model.parameters())
    model = prune_yolo11n_p5(model, PROFILES[PROFILE_NAME])
    after = sum(parameter.numel() for parameter in model.parameters())

    with torch.no_grad():
        output = model(torch.zeros(1, 3, 640, 640))
    scores = output[1]["scores"] if isinstance(output, tuple) else output["scores"]
    boxes = output[1]["boxes"] if isinstance(output, tuple) else output["boxes"]

    print("profile", PROFILE_NAME)
    print("params_before", before)
    print("params_after", after)
    print("reduction_percent", 100.0 * (before - after) / before)
    print("score_shape", tuple(scores.shape))
    print("box_shape", tuple(boxes.shape))
    for index in (7, 8, 9, 10, 20, 22):
        print(index, model.model[index])


if __name__ == "__main__":
    main()
