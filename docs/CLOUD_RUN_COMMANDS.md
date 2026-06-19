# Cloud Run Commands

This file is the recommended execution order for the cloud machine. Do not skip the build checks before training.

## 1. Install

```bash
cd grape-yolo11-efsa-disease-detection
python -m pip install -U pip
pip install -e .
pip install pytest
```

If the cloud image does not include PyTorch, install a CUDA-matched PyTorch wheel first, then run `pip install -e .`.

## 2. Point Dataset YAML To The Uploaded Data

Edit `configs/data/grape.yaml` and `configs/data/grape_full.yaml` if the dataset is not placed next to the repository.

The expected YOLO directory structure is:

```text
dataset_root/
  train/images
  train/labels
  val/images
  val/labels
  test/images
  test/labels
```

## 3. Structural Checks

```bash
pytest -q tests/test_yolo11_efsa_build.py
```

```bash
python - <<'PY'
from ultralytics import YOLO
m = YOLO('configs/models/yolo11n_efsa_grape.yaml')
m.info()
PY
```

## 4. Baseline Training

Run a clean YOLO11 baseline first. This is required for a defensible paper comparison.

```bash
python scripts/train_baseline.py \
  --data configs/data/grape.yaml \
  --imgsz 768 \
  --epochs 150 \
  --batch 8 \
  --device 0 \
  --name yolo11n_baseline_768_seed42
```

## 5. Grape-EFSA-YOLO11 Training

```bash
python scripts/train_efsa.py \
  --data configs/data/grape.yaml \
  --model configs/models/yolo11n_efsa_grape.yaml \
  --pretrained yolo11n.pt \
  --imgsz 768 \
  --epochs 150 \
  --batch 8 \
  --device 0 \
  --name yolo11n_efsa_768_seed42
```

## 6. Ablation Training

```bash
python scripts/train_efsa.py --model configs/ablations/yolo11n_edge_small_grape.yaml --name ablate_edge_small
python scripts/train_efsa.py --model configs/ablations/yolo11n_carafe_bifpn_grape.yaml --name ablate_carafe_bifpn
python scripts/train_efsa.py --model configs/ablations/yolo11n_attention_grape.yaml --name ablate_attention_full
```

## 7. Validation

```bash
python scripts/val.py \
  --weights runs/grape/yolo11n_efsa_768_seed42/weights/best.pt \
  --data configs/data/grape.yaml \
  --imgsz 768 \
  --device 0
```

## 8. Speed And Export

```bash
python scripts/benchmark_fps.py --weights runs/grape/yolo11n_efsa_768_seed42/weights/best.pt --imgsz 768 --device 0
python scripts/export_onnx.py --weights runs/grape/yolo11n_efsa_768_seed42/weights/best.pt --imgsz 768
```
