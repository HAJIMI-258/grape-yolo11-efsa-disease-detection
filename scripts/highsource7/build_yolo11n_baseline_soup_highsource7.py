from __future__ import annotations

import csv
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import torch

REMOTE = Path(r"D:\grape_combo")
REPO = REMOTE / "grape-yolo11-efsa-disease-detection"
if REPO.exists():
    sys.path.insert(0, str(REPO))

from ultralytics import YOLO
from ultralytics.utils import LOGGER

DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")
MAP_WEIGHTS = Path(r"D:\grape_combo\runs\yolo11n_highsource7_region_fast_img640_e150\weights\best.pt")
AP_WEIGHTS = Path(
    r"D:\grape_combo\runs\yolo11n_highsource7_region_ap50teacher_img640_e150\weights\best_map50.pt"
)
OUTPUT_DIR = REMOTE / "baseline_soup"
BASELINE_AP50 = 0.96878


def _alphas() -> list[float]:
    raw = os.environ.get("GRAPE_SOUP_ALPHAS", "0,0.25,0.5,0.75,1")
    values = sorted({float(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"Invalid GRAPE_SOUP_ALPHAS={raw!r}")
    return values


def _interpolate_models(map_model, ap_model, alpha: float):
    soup = deepcopy(map_model).float().cpu()
    map_state = map_model.float().cpu().state_dict()
    ap_state = ap_model.float().cpu().state_dict()
    soup_state = soup.state_dict()

    if map_state.keys() != ap_state.keys():
        raise RuntimeError("Baseline checkpoint architectures do not match")

    blended = {}
    for key, target in soup_state.items():
        left = map_state[key]
        right = ap_state[key]
        if left.shape != right.shape or left.shape != target.shape:
            raise RuntimeError(f"Checkpoint tensor mismatch at {key}: {left.shape}, {right.shape}, {target.shape}")
        if target.dtype.is_floating_point:
            blended[key] = left.to(torch.float32).mul(1.0 - alpha).add(right.to(torch.float32), alpha=alpha)
        else:
            blended[key] = right if alpha >= 0.5 else left
    soup.load_state_dict(blended, strict=True)
    return soup


def main() -> None:
    if not MAP_WEIGHTS.exists() or not AP_WEIGHTS.exists():
        raise FileNotFoundError(f"Soup inputs missing: {MAP_WEIGHTS}, {AP_WEIGHTS}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    map_wrapper = YOLO(str(MAP_WEIGHTS))
    ap_wrapper = YOLO(str(AP_WEIGHTS))
    map_model = map_wrapper.model
    ap_model = ap_wrapper.model
    rows: list[dict[str, float | str]] = []

    for alpha in _alphas():
        candidate = _interpolate_models(map_model, ap_model, alpha)
        candidate_path = OUTPUT_DIR / f"yolo11n_highsource7_soup_a{alpha:.2f}.pt"
        wrapper = YOLO(str(MAP_WEIGHTS))
        wrapper.model = candidate
        wrapper.save(candidate_path)
        metrics = YOLO(str(candidate_path)).val(
            data=str(DATA_CFG),
            imgsz=640,
            batch=64,
            workers=4,
            device=0,
            plots=False,
            verbose=False,
            project=str(REMOTE / "runs"),
            name=f"yolo11n_highsource7_soup_a{alpha:.2f}_val",
        )
        row = {
            "alpha_ap_checkpoint": alpha,
            "ap50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "checkpoint": str(candidate_path),
        }
        rows.append(row)
        LOGGER.info("Soup alpha=%.2f: AP50=%.5f, mAP50-95=%.5f", alpha, row["ap50"], row["map50_95"])

    rows.sort(key=lambda item: (float(item["ap50"]), float(item["map50_95"])), reverse=True)
    best = rows[0]
    best_path = OUTPUT_DIR / "yolo11n_highsource7_soup_best.pt"
    shutil.copy2(Path(str(best["checkpoint"])), best_path)

    with (OUTPUT_DIR / "soup_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alpha_ap_checkpoint", "ap50", "map50_95", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Best soup: %s", best)
    if float(best["ap50"]) <= BASELINE_AP50:
        LOGGER.warning(
            "Soup did not exceed baseline AP50 %.5f. Do not use it as GRAPE_PRUNE_SOURCE; prune the AP50 checkpoint.",
            BASELINE_AP50,
        )
    else:
        LOGGER.info(
            "Soup exceeds baseline. Use: $env:GRAPE_PRUNE_SOURCE='%s'",
            best_path,
        )


if __name__ == "__main__":
    main()
