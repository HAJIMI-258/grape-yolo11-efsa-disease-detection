from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(r"D:\grape_combo\grape-yolo11-efsa-disease-detection")
if REPO.exists():
    sys.path.insert(0, str(REPO))

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

TEACHER_WEIGHTS = Path(r"D:\grape_combo\runs\yolo11n_highsource7_region_fast_img640_e150\weights\best.pt")
STUDENT_WEIGHTS = Path(r"D:\grape_mypfe6\yolo11n.pt")
DATA_CFG = Path(r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml")

CLASS_NAMES = [
    "black_rot",
    "esca_black_measles",
    "healthy",
    "leaf_blight",
    "brown_spot",
    "downy_mildew",
    "mites_disease",
]

# Mild weights only. The full YOLO11n baseline is already strong, so this is
# intended as a small positive-region guide, not an aggressive rebalancing loss.
CLASS_KD_WEIGHTS = [1.00, 0.90, 1.08, 0.90, 1.22, 1.12, 1.22]


def _select_o2m(preds):
    if isinstance(preds, tuple):
        preds = preds[1]
    if isinstance(preds, dict) and "one2many" in preds:
        return preds["one2many"]
    return preds


class FullGapDistillCriterion:
    """YOLO11n base loss plus conservative foreground teacher guidance."""

    def __init__(
        self,
        student,
        teacher,
        cls_weight: float = 0.35,
        response_weight: float = 0.12,
        dfl_weight: float = 0.03,
        temperature: float = 3.0,
        bg_weight: float = 0.01,
        fg_threshold: float = 0.30,
    ):
        self.base = student.init_criterion()
        self.teacher = teacher
        self.cls_weight = cls_weight
        self.response_weight = response_weight
        self.dfl_weight = dfl_weight
        self.temperature = temperature
        self.bg_weight = bg_weight
        self.fg_threshold = fg_threshold
        self.class_weights = torch.tensor(CLASS_KD_WEIGHTS, dtype=torch.float32)

    def __call__(self, preds, batch):
        base_loss, loss_items = self.base(preds, batch)
        student_preds = _select_o2m(preds)
        if not torch.is_grad_enabled() or not isinstance(student_preds, dict):
            return base_loss, loss_items
        if "scores" not in student_preds or not student_preds["scores"].requires_grad:
            return base_loss, loss_items

        with torch.no_grad():
            head = self.teacher.model[-1]
            was_training = head.training
            head.training = True
            teacher_preds = self.teacher(batch["img"])
            head.training = was_training
            teacher_preds = _select_o2m(teacher_preds)

        kd = self._head_distill_loss(student_preds, teacher_preds)
        return base_loss + kd * batch["img"].shape[0], loss_items

    def _head_distill_loss(self, student_preds, teacher_preds):
        if not isinstance(teacher_preds, dict):
            return torch.zeros((), device=student_preds["scores"].device)
        if student_preds["scores"].shape != teacher_preds["scores"].shape:
            return torch.zeros((), device=student_preds["scores"].device)

        t = self.temperature
        teacher_scores = teacher_preds["scores"].detach()
        student_scores = student_preds["scores"]
        class_weights = self.class_weights.to(student_scores.device).view(1, -1, 1)

        teacher_prob = teacher_scores.sigmoid()
        anchor_conf, anchor_cls = teacher_prob.max(dim=1, keepdim=True)
        anchor_weight = self.bg_weight + anchor_conf.pow(2)

        cls_target = (teacher_scores / t).sigmoid()
        cls_loss = F.binary_cross_entropy_with_logits(student_scores / t, cls_target, reduction="none") * (t * t)
        cls_loss = (cls_loss * anchor_weight * class_weights).mean()

        selected_logits = student_scores.gather(1, anchor_cls)
        expanded_class_weights = class_weights.expand(student_scores.shape[0], -1, student_scores.shape[2])
        selected_class_weight = expanded_class_weights.gather(1, anchor_cls)
        fg_mask = anchor_conf.gt(self.fg_threshold).float()
        response_loss = F.binary_cross_entropy_with_logits(selected_logits, anchor_conf, reduction="none")
        response_loss = (response_loss * fg_mask * selected_class_weight).sum() / fg_mask.sum().clamp_min(1.0)

        dfl_loss = self._dfl_distill_loss(student_preds, teacher_preds, anchor_weight)
        return self.cls_weight * cls_loss + self.response_weight * response_loss + self.dfl_weight * dfl_loss

    def _dfl_distill_loss(self, student_preds, teacher_preds, anchor_weight):
        student_boxes = student_preds["boxes"]
        teacher_boxes = teacher_preds["boxes"].detach()
        if student_boxes.shape != teacher_boxes.shape or student_boxes.shape[1] % 4 != 0:
            return torch.zeros((), device=student_preds["scores"].device)

        t = self.temperature
        batch, channels, anchors = student_boxes.shape
        reg_max = channels // 4
        s_dist = student_boxes.view(batch, 4, reg_max, anchors).permute(0, 1, 3, 2).contiguous()
        t_dist = teacher_boxes.view(batch, 4, reg_max, anchors).permute(0, 1, 3, 2).contiguous()
        dfl_loss = F.kl_div(
            F.log_softmax(s_dist / t, dim=-1),
            F.softmax(t_dist / t, dim=-1),
            reduction="none",
        ).sum(-1) * (t * t)
        return (dfl_loss.mean(1) * anchor_weight.squeeze(1)).mean()


class FullGapDistillTrainer(DetectionTrainer):
    """Attach the fixed full YOLO11n baseline teacher during training only."""

    def set_model_attributes(self):
        super().set_model_attributes()
        teacher = YOLO(str(TEACHER_WEIGHTS)).model.to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher
        self.model.criterion = FullGapDistillCriterion(self.model, teacher)
        LOGGER.info(
            "Full YOLO11n GapKD enabled: teacher=%s, cls_weight=0.35, "
            "response_weight=0.12, dfl_weight=0.03, T=3.0, bg_weight=0.01, fg_threshold=0.30",
            TEACHER_WEIGHTS,
        )
        LOGGER.info("Class KD weights: %s", dict(zip(CLASS_NAMES, CLASS_KD_WEIGHTS)))


def main() -> None:
    model = YOLO(str(STUDENT_WEIGHTS))
    model.train(
        trainer=FullGapDistillTrainer,
        data=str(DATA_CFG),
        imgsz=640,
        epochs=150,
        batch=64,
        workers=4,
        device=0,
        pretrained=True,
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
        name="yolo11n_gapkd_full_highsource7_region_img640_e150",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
