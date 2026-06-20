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

from p3p4_compaction import compact_p251_to_p3p4, save_compact_checkpoint

from ultralytics import YOLO

DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")
SOURCE = Path(
    os.environ.get(
        "GRAPE_P251_SOURCE",
        r"D:\grape_combo\runs\yolo11n_p5prune_v20_p251_highsource7_region_img640_e150\weights\best_map50.pt",
    )
)
OUTPUT = REMOTE / "compact_models" / "yolo11n_p251_p3p4compact_v21_before_recovery.pt"


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    source = YOLO(str(SOURCE)).model.float().cpu().eval()
    before = sum(parameter.numel() for parameter in source.parameters())
    compact = compact_p251_to_p3p4(source)
    after = sum(parameter.numel() for parameter in compact.parameters())
    save_compact_checkpoint(compact, SOURCE, OUTPUT)

    summary = YOLO(str(OUTPUT)).info(imgsz=640, verbose=False)
    metrics = YOLO(str(OUTPUT)).val(
        data=str(DATA_CFG),
        imgsz=640,
        batch=64,
        workers=4,
        device=0,
        plots=False,
        verbose=False,
        project=str(REMOTE / "runs"),
        name="yolo11n_p3p4compact_v21_dryrun_val",
    )

    print("source_params", before)
    print("compact_params", after)
    print("parameter_reduction_percent", 100.0 * (before - after) / before)
    print("model_info", summary)
    print("precision", float(metrics.box.mp))
    print("recall", float(metrics.box.mr))
    print("ap50", float(metrics.box.map50))
    print("map50_95", float(metrics.box.map))
    print("checkpoint", OUTPUT)


if __name__ == "__main__":
    main()
