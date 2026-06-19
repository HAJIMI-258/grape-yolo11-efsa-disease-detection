from __future__ import annotations

import os
import sys
from pathlib import Path

REMOTE = Path(r"D:\grape_combo")
REPO = REMOTE / "grape-yolo11-efsa-disease-detection"
if REPO.exists():
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REMOTE))

from highsource7_ap50_checkpoint import AP50CheckpointMixin
from train_yolo11n_width20_gapkd_v12_highsource7_region_remote import CLASS_KD_WEIGHTS, CLASS_NAMES
from train_yolo11n_width20_gapkd_v14_highsource7_region_remote import LateFeatureGapCriterion

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")
COCO_INIT = Path(r"D:\grape_mypfe6\yolo11n.pt")
TEACHER_WEIGHTS = Path(r"D:\grape_combo\runs\yolo11n_highsource7_region_fast_img640_e150\weights\best.pt")


class GapKDV18Trainer(AP50CheckpointMixin, DetectionTrainer):
    """Use the proven v14 KD schedule for the parameter-matched 1.4M students."""

    def set_model_attributes(self):
        super().set_model_attributes()
        if not TEACHER_WEIGHTS.exists():
            raise FileNotFoundError(f"Frozen YOLO11n teacher not found: {TEACHER_WEIGHTS}")

        teacher = YOLO(str(TEACHER_WEIGHTS)).model.to(self.device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher = teacher
        self.model.criterion = LateFeatureGapCriterion(
            self.model,
            teacher,
            cls_weight=0.70,
            response_weight=0.18,
            dfl_weight=0.06,
            temperature=3.0,
            bg_weight=0.03,
            fg_threshold=0.25,
            gt_expand=1.40,
            feature_start_epoch=80,
            feature_ramp_epochs=30,
            feature_final_weight=0.025,
        )
        LOGGER.info(
            "V18 GapKD enabled: teacher=%s, late P3/P4 foreground KD start=80 ramp=30 final=0.025",
            TEACHER_WEIGHTS,
        )
        LOGGER.info("Class KD weights: %s", dict(zip(CLASS_NAMES, CLASS_KD_WEIGHTS)))

    def run_callbacks(self, event: str):
        if event == "on_train_epoch_start" and hasattr(self.model, "criterion"):
            criterion = self.model.criterion
            if hasattr(criterion, "current_epoch"):
                criterion.current_epoch = int(self.epoch) + 1
        return super().run_callbacks(event)


def train_v18(model_cfg: Path, run_name: str, *, wiou_alpha: float = 0.0) -> None:
    """Train one v18 variant under the locked HighSource7 protocol."""
    if not model_cfg.exists():
        raise FileNotFoundError(f"Model config not found: {model_cfg}")
    if not COCO_INIT.exists():
        raise FileNotFoundError(f"YOLO11n COCO initialization not found: {COCO_INIT}")

    # BboxLoss reads this value when the criterion is constructed.
    os.environ["YOLO_WIOU_ALPHA"] = str(float(wiou_alpha))
    os.environ["YOLO_NWD_ALPHA"] = "0.0"

    model = YOLO(str(model_cfg))
    # This is fair to the baseline: transfer only exact-shape COCO tensors.
    # The v18 channel plan deliberately preserves YOLO11n layers 0-6.
    model.load(str(COCO_INIT))
    summary = model.info(imgsz=640, verbose=False)
    if summary:
        _, parameters, _, gflops = summary
        LOGGER.info("V18 deployment model: params=%d, GFLOPs=%.3f", parameters, gflops)
        if not 1_350_000 <= parameters <= 1_450_000:
            raise RuntimeError(f"V18 model is outside the 1.4M budget: {parameters:,} parameters")

    model.train(
        trainer=GapKDV18Trainer,
        data=str(DATA_CFG),
        imgsz=640,
        epochs=150,
        batch=64,
        workers=4,
        device=0,
        pretrained=False,
        patience=0,
        amp=True,
        cache=False,
        seed=42,
        box=8.0,
        cls=0.65,
        dfl=1.7,
        mosaic=0.45,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=20,
        project=r"D:\grape_combo\runs",
        name=run_name,
        exist_ok=True,
        verbose=True,
    )
