# Grape-EFSA-YOLO11 方法说明

## 定位

Grape-EFSA-YOLO11 是面向葡萄病害检测的 YOLO11 结构增强方法。它不是 RT-DETR，也不是把 D-FINE decoder 强塞进 YOLO11，而是在 YOLO11 的 P3/P4/P5 多尺度检测路径上迁移边缘、小病斑、多尺度融合和轻量注意力思想。

## 从 RT-DETR v6 迁移的部分

- 边缘引导增强：用于提高病斑边界、叶片坏死边缘和虫害损伤轮廓响应。
- 小病斑跨尺度增强：使用 3x3、5x5、7x7 depthwise 分支提取不同尺度纹理。
- 输出细化：对 P3/P4/P5 特征做局部上下文残差细化。
- CARAFE-style 内容感知上采样：替代普通最近邻上采样，减少小病斑边界模糊。
- BiFPN-style 加权融合：使用可学习非负权重融合跨尺度特征。
- Coordinate Attention + ECA：同时建模位置关系和通道判别性。
- 安全残差门控：所有新增增强以有上界残差注入，初始扰动较小。

## 没有迁移的部分

- D-FINE decoder：属于 RT-DETR 查询式 decoder，与 YOLO11 anchor-free Detect head 不兼容。
- Hungarian matcher：用于 DETR 类集合预测训练，不适合直接替代 YOLO11 的正负样本分配流程。
- RTv4Criterion / MAL / FGL / DDF：这些损失项依赖 RT-DETR/D-FINE 的预测结构，第一版不迁移。
- DINOv3 foreground distillation：需要 teacher 特征和蒸馏训练管线，放到后续分支。
- NWD：可以作为小目标辅助损失后续单独做 `feature/nwd-loss`，第一版先保持 YOLO11 原生检测 loss。

## 模块

### EFSAEnhance

输入输出保持 `B,C,H,W -> B,C,H,W`。

增强分支：

```text
enhancement =
  edge_alpha   * PrewittHighFrequency(x)
+ lesion_alpha * MultiKernelDWConv3x5x7(x)
+ refine_alpha * LocalRefine(x)
```

输出：

```text
scale = max_scale * tanh(raw_scale)
y = x + scale * enhancement
```

### CARAFEUp

纯 PyTorch 内容感知上采样：

```text
compress 1x1 -> kernel encoder -> pixel shuffle -> softmax weights
unfold low-res neighborhoods -> nearest align neighborhoods -> weighted reassembly
```

不依赖 MMCV，输出尺寸必须与 `F.interpolate(scale_factor=2)` 一致。

### BiFPNFuse

对每个输入特征做 1x1 projection 到同一通道数，再用可学习非负归一化权重融合：

```text
w_i = relu(raw_w_i) / (sum(relu(raw_w)) + eps)
y = sum_i w_i * proj_i(resize_i(x_i))
```

### CoordECA

Coordinate Attention 保留高度/宽度方向位置信息，ECA 使用全局平均池化和 1D 卷积做轻量通道重标定，最后通过安全残差门控注入。

## YOLO11 结构路径

```text
YOLO11 backbone
  -> EFSAEnhance on P3/P4/P5
  -> CARAFEUp top-down path
  -> BiFPNFuse weighted cross-scale fusion
  -> CoordECA before Detect
  -> YOLO11 Detect head
```

## 实验原则

- 先跑 clean YOLO11 baseline，再跑 EFSA。
- 所有实验固定数据划分、输入尺寸、epoch、batch、seed。
- 报告 Precision、Recall、AP50、mAP50:95、AP75、per-class AP、Params、FLOPs、FPS。
- 棉花项目结果不能写成葡萄实验结果。
