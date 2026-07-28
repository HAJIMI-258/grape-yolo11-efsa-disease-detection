# Grape-EFSA-YOLO11

YOLO11-based grape disease detection with edge-guided fine-grained small-lesion attention modules.

This repository is built from Ultralytics YOLO11 source code and keeps the original YOLO11 detection pipeline: C3k2, SPPF, C2PSA, and Detect head. It adds grape-specific feature enhancement modules for small lesion boundaries and multi-scale disease patterns.

## Method

```text
YOLO11 backbone
  -> EFSAEnhance(P3/P4/P5)
  -> CARAFEUp
  -> BiFPNFuse
  -> CoordECA
  -> YOLO11 Detect
```

Migrated from the RT-DETR v6 method line:

- Edge-guided enhancement
- Small-lesion multi-scale depthwise enhancement
- Output refinement
- CARAFE-style upsampling
- BiFPN-style weighted fusion
- Coordinate Attention
- ECA channel attention
- Safe residual gate

Not migrated in the first version:

- D-FINE decoder
- Hungarian matcher
- RTv4Criterion
- MAL/FGL/DDF losses
- DINOv3 foreground distillation

The first version keeps native YOLO11 detection loss. NWD small-object loss should be added later in a separate branch after the structure is stable.

## Data

Default configs:

```text
configs/data/grape.yaml       # balanced_fine_yolo, recommended for quick experiments
configs/data/grape_full.yaml  # merged_fine_yolo, recommended for final full-data training
```

The current class list has 11 classes:

```text
black_rot
downy_mildew
esca_black_measles
leaf_blight
powdery_mildew
dead_arm
anthracnose
flavescence_doree
healthy
brown_spot
mites_disease
```

## Quick Start

Recommended cloud execution order:

```text
docs/CLOUD_RUN_COMMANDS.md
```

Install on the cloud machine:

```bash
pip install -e .
pip install pytest
```

Build check:

```bash
python - << 'PY'
from ultralytics import YOLO
m = YOLO('configs/models/yolo11n_efsa_grape.yaml')
m.info()
PY
```

Train baseline:

```bash
python scripts/train_baseline.py --data configs/data/grape.yaml --imgsz 768 --epochs 150 --batch 8 --device 0
```

Train EFSA:

```bash
python scripts/train_efsa.py --data configs/data/grape.yaml --imgsz 768 --epochs 150 --batch 8 --device 0
```

Validate:

```bash
python scripts/val.py --weights runs/grape/yolo11n_efsa_768_seed42/weights/best.pt --data configs/data/grape.yaml
```

Benchmark FPS:

```bash
python scripts/benchmark_fps.py --weights runs/grape/yolo11n_efsa_768_seed42/weights/best.pt --imgsz 768
```

Export ONNX:

```bash
python scripts/export_onnx.py --weights runs/grape/yolo11n_efsa_768_seed42/weights/best.pt --imgsz 768
```

## Experiments

See:

- `docs/METHOD_Grape_EFSA_YOLO11.md`
- `docs/EXPERIMENT_CHECKLIST.md`
- `docs/ABLATION_PLAN.md`
- `docs/CLOUD_RUN_COMMANDS.md`

Report at least:

| Model              | Params | FLOPs | FPS | Precision | Recall | AP50 | AP75 | mAP50:95 |
| ------------------ | -----: | ----: | --: | --------: | -----: | ---: | ---: | -------: |
| YOLO11n baseline   |      - |     - |   - |         - |      - |    - |    - |        - |
| Grape-EFSA-YOLO11n |      - |     - |   - |         - |      - |    - |    - |        - |

## License

This repository modifies Ultralytics YOLO11 source code and therefore keeps the Ultralytics AGPL-3.0 license unless a separate Ultralytics Enterprise license is obtained.
