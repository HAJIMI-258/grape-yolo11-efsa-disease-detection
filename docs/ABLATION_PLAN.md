# 消融实验计划

| 实验 | 配置 | 目的 |
|---|---|---|
| Baseline | `configs/ablations/yolo11n_baseline_grape.yaml` 或 `yolo11n.pt` | 干净 YOLO11 对照 |
| + EFSAEnhance | `configs/ablations/yolo11n_edge_small_grape.yaml` | 验证边缘/小病斑增强 |
| + CARAFE/BiFPN | `configs/ablations/yolo11n_carafe_bifpn_grape.yaml` | 验证内容感知上采样和加权融合 |
| Full EFSA | `configs/ablations/yolo11n_attention_grape.yaml` | 验证 CoordECA 加入后的完整结构 |
| Full EFSA + NWD | 后续 `feature/nwd-loss` 分支 | 验证小目标定位辅助损失 |

所有实验必须使用相同：

- 数据划分
- 输入尺寸
- epoch
- batch 或梯度累积策略
- seed
- 预训练策略

论文表格建议列：

| Model | Params | FLOPs | FPS | Precision | Recall | AP50 | AP75 | mAP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
