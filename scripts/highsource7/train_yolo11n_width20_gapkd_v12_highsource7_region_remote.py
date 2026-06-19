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
STUDENT_CFG = Path(r"D:\grape_combo\yolo11n_width20_highsource7.yaml")
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

# Keep the v1 class-gap guidance because it is the only 1.2M run that recovered
# AP50 well. v1.2 only adds low-weight foreground P3/P4 feature alignment.
CLASS_KD_WEIGHTS = [1.00, 1.00, 1.05, 1.00, 1.45, 1.12, 1.50]


def _select_o2m(preds):
    """Return the training prediction dict used for one-to-many detection loss."""
    if isinstance(preds, tuple):
        preds = preds[1]
    if isinstance(preds, dict) and "one2many" in preds:
        return preds["one2many"]
    return preds


class GapFeatureDistillCriterion:
    """V1 head KD plus foreground-only P3/P4 spatial feature distillation."""

    def __init__(
        self,
        student,
        teacher,
        cls_weight: float = 0.70,
        response_weight: float = 0.18,
        dfl_weight: float = 0.06,
        feature_weight: float = 0.06,
        temperature: float = 3.0,
        bg_weight: float = 0.03,
        fg_threshold: float = 0.25,
        gt_expand: float = 1.40,
    ):
        self.base = student.init_criterion()
        self.teacher = teacher
        self.cls_weight = cls_weight
        self.response_weight = response_weight
        self.dfl_weight = dfl_weight
        self.feature_weight = feature_weight
        self.temperature = temperature
        self.bg_weight = bg_weight
        self.fg_threshold = fg_threshold
        self.gt_expand = gt_expand
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
        feat_kd = self._foreground_feature_loss(student_preds, teacher_preds, batch)
        return base_loss + (kd + self.feature_weight * feat_kd) * batch["img"].shape[0], loss_items

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

    def _foreground_feature_loss(self, student_preds, teacher_preds, batch):
        s_feats = student_preds.get("feats") if isinstance(student_preds, dict) else None
        t_feats = teacher_preds.get("feats") if isinstance(teacher_preds, dict) else None
        if not s_feats or not t_feats:
            return torch.zeros((), device=student_preds["scores"].device)

        total = torch.zeros((), device=s_feats[0].device)
        used = 0
        # P3/P4 only: shallow and middle features carry lesion color, edge and texture cues.
        for idx, level_weight in ((0, 0.60), (1, 0.40)):
            if idx >= len(s_feats) or idx >= len(t_feats):
                continue
            s_feat = s_feats[idx]
            t_feat = t_feats[idx].detach()
            mask = self._gt_feature_mask(s_feat, batch, idx)
            if mask.sum() <= 0:
                continue

            s_attn = self._spatial_attention(s_feat)
            t_attn = self._spatial_attention(t_feat)
            if t_attn.shape[-2:] != s_attn.shape[-2:]:
                t_attn = F.interpolate(t_attn, size=s_attn.shape[-2:], mode="bilinear", align_corners=False)
            diff = (s_attn - t_attn).pow(2) * mask
            total = total + level_weight * diff.sum() / mask.sum().clamp_min(1.0)
            used += 1
        return total / max(used, 1)

    @staticmethod
    def _spatial_attention(feat):
        attn = feat.float().pow(2).mean(1, keepdim=True)
        flat = attn.flatten(2)
        flat = F.normalize(flat, p=2, dim=2)
        return flat.view_as(attn)

    def _gt_feature_mask(self, feat, batch, level_idx):
        device = feat.device
        dtype = feat.dtype
        b, _, h, w = feat.shape
        if batch["bboxes"].numel() == 0:
            return torch.zeros((b, 1, h, w), device=device, dtype=dtype)

        # Reuse Ultralytics preprocessing so mosaic/letterbox training boxes are
        # interpreted exactly like the base detection loss.
        imgsz = torch.tensor([h, w], device=device, dtype=dtype) * self.base.stride[level_idx]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.base.preprocess(targets.to(device), b, scale_tensor=imgsz[[1, 0, 1, 0]])
        boxes = targets[..., 1:5]
        valid = boxes.sum(2).gt(0)
        if not valid.any():
            return torch.zeros((b, 1, h, w), device=device, dtype=dtype)

        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = (x2 - x1).clamp_min(1.0) * self.gt_expand
        bh = (y2 - y1).clamp_min(1.0) * self.gt_expand
        x1 = (cx - bw * 0.5).clamp_min(0)
        y1 = (cy - bh * 0.5).clamp_min(0)
        x2 = (cx + bw * 0.5).clamp_max(imgsz[1])
        y2 = (cy + bh * 0.5).clamp_max(imgsz[0])

        stride = self.base.stride[level_idx].to(device=device, dtype=dtype)
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xx = xx.view(1, 1, h, w)
        yy = yy.view(1, 1, h, w)

        inside = (
            (xx >= x1.unsqueeze(-1).unsqueeze(-1))
            & (xx <= x2.unsqueeze(-1).unsqueeze(-1))
            & (yy >= y1.unsqueeze(-1).unsqueeze(-1))
            & (yy <= y2.unsqueeze(-1).unsqueeze(-1))
            & valid.unsqueeze(-1).unsqueeze(-1)
        )
        return inside.any(dim=1, keepdim=True).to(dtype=dtype)


class GapFeatureDistillTrainer(DetectionTrainer):
    """Attach a frozen teacher to the student model after it has moved to device."""

    def set_model_attributes(self):
        super().set_model_attributes()
        teacher = YOLO(str(TEACHER_WEIGHTS)).model.to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self.teacher = teacher
        self.model.criterion = GapFeatureDistillCriterion(self.model, teacher)
        LOGGER.info(
            "GapKD v1.2 enabled: teacher=%s, v1 head KD + foreground P3/P4 feature KD, "
            "feature_weight=0.06, gt_expand=1.40, no hard negatives",
            TEACHER_WEIGHTS,
        )
        LOGGER.info("Class KD weights: %s", dict(zip(CLASS_NAMES, CLASS_KD_WEIGHTS)))


def main() -> None:
    model = YOLO(str(STUDENT_CFG))
    model.train(
        trainer=GapFeatureDistillTrainer,
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
        name="yolo11n_width20_gapkd_v12_highsource7_region_img640_e150",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
