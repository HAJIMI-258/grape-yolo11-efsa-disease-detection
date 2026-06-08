# 实验检查清单

## 数据

- 使用同一份 `configs/data/grape.yaml` 或 `configs/data/grape_full.yaml`。
- 不手工混合 train/val/test。
- 确认类别数为 11，类别名与数据集 `data.yaml` 一致。
- 先用 `balanced_fine_yolo` 快速验证结构，再用 `merged_fine_yolo` 做全量最终训练。

## 训练

- Baseline：`scripts/train_baseline.py`
- EFSA：`scripts/train_efsa.py`
- 固定 `imgsz=768`、`seed=42`、相同 epoch、batch、workers。
- 记录显存、训练时间、最佳 epoch。

## 指标

- Precision
- Recall
- AP50
- mAP50:95
- AP75
- per-class AP
- Params
- FLOPs
- FPS / latency

## 风险项

- 如果小类 AP 波动大，优先检查样本量和标注一致性。
- 如果 EFSA 训练不稳定，先降低学习率或把 `max_scale` 从 0.10 降到 0.05。
- 如果 FPS 降太多，先消融 CARAFEUp，再消融 CoordECA。
