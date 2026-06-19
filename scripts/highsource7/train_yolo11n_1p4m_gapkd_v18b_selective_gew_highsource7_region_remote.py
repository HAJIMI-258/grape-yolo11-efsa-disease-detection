from __future__ import annotations

import sys
from pathlib import Path

REMOTE = Path(r"D:\grape_combo")
REPO = REMOTE / "grape-yolo11-efsa-disease-detection"
if REPO.exists():
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REMOTE))

from train_yolo11n_1p4m_gapkd_v18_common import train_v18

MODEL_CFG = REPO / "configs/models/yolo11n_1p4m_selective_gew_highsource7_region.yaml"
RUN_NAME = "yolo11n_1p4m_gapkd_v18b_selective_gew_highsource7_region_img640_e150"


if __name__ == "__main__":
    train_v18(MODEL_CFG, RUN_NAME)
