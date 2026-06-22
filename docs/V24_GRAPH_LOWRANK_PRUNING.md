# V24: Graph-Aware Rank-Channel Pruning on Low-Rank YOLO11

## Purpose

V23 already produced the current sub-2M endpoint:

```text
D:\grape_combo\lowrank_models\yolo11n_v23_p1990k_dryrun.pt
Params: 1.990M
AP50:   0.97450
```

V24 is not a replacement for this result yet. It tests whether further compression can remove **interaction channels inside
LowRankConv blocks** without changing external feature widths, Detect scales, or the 8400-anchor output layout.

## Method

V24 applies Torch-Pruning dependency groups to the internal rank dimension of V23 `LowRankConv` modules:

```text
LowRankConv:
  spatial Conv(c1 -> r)
  pointwise Conv(r -> c2)

V24 removes selected rank channels:
  spatial out channel j
  pointwise input channel j
```

The external `c2` output is unchanged. This means P3/P4/P5 shapes, Detect heads, and anchor ordering remain compatible with
the original YOLO11 validator.

Rank-channel importance is computed from both sides of the factorization:

```text
importance(j) = ||spatial_j|| * ||pointwise_:,j||
```

If gradients are present, the implementation adds a Taylor/Fisher proxy:

```text
sqrt(sum((w * grad)^2))
```

Current dry runs used the weight-only form. P3/P4 classification towers and shallow layers are protected.

## Current Dry-Run Results

All runs below are explicit validation runs on the fixed HighSource7 validation split at `imgsz=640`.

| Source checkpoint | Target |   Params | GFLOPs | Precision |  Recall |      AP50 |   AP50-95 | Decision                                          |
| ----------------- | -----: | -------: | -----: | --------: | ------: | --------: | --------: | ------------------------------------------------- |
| V23 p1990 dry-run |  1.95M | `1.944M` |  `5.7` |   `0.942` | `0.945` | `0.97337` | `0.81951` | Smaller than V23 p1990, AP50 still above baseline |
| V23 p1990 dry-run |  1.90M | `1.899M` |  `5.6` |   `0.944` | `0.941` | `0.97065` | `0.81551` | More compression, but AP50 drop becomes visible   |
| V23 p2200 trained |  1.99M | `1.988M` |  `5.8` |   `0.955` | `0.928` | `0.96868` | `0.80767` | Rejected; direct p2200 graph-prune hurts recall   |

Reference points:

| Model             |   Params |      AP50 |   AP50-95 |
| ----------------- | -------: | --------: | --------: |
| YOLO11n baseline  | `2.591M` | `0.96878` | `0.82287` |
| V23 p1990 dry-run | `1.990M` | `0.97450` | `0.81759` |

## Interpretation

V24 confirms that graph-aware rank pruning is a real additional compression lever, but the current policy does **not** improve
over the V23 p1990 AP50 endpoint. It creates a smaller valid candidate at `1.944M / AP50=0.97337`, but the best sub-2M model
remains V23 p1990 dry-run.

The p2200-to-p1990 result is especially important: starting from the highest-AP50 trained checkpoint and graph-pruning directly
to 2M loses recall and falls to the baseline boundary. This suggests the successful p1990 result depends on the progressive
p2100-dryrun-to-p1990 route, not just the final parameter count.

## Commands

Conservative dry run from V23 p1990:

```powershell
$env:PYTHONPATH="D:\grape_combo\grape-yolo11-efsa-disease-detection;D:\grape_combo\grape-yolo11-efsa-disease-detection\scripts\highsource7;D:\grape_combo"
$env:GRAPE_V24_SOURCE="D:\grape_combo\lowrank_models\yolo11n_v23_p1990k_dryrun.pt"
$env:GRAPE_V24_TAG="from_v23_p1990"
$env:GRAPE_V24_TARGET="1950000"
$env:GRAPE_V24_PRUNE_STEP="4"
$env:GRAPE_V24_MIN_RANK="8"
$env:GRAPE_V24_MIN_IMPORTANCE="0.95"
D:\Python311\python.exe scripts\highsource7\dryrun_yolo11n_graphlowrank_v24_remote.py
```

Aggressive dry run:

```powershell
$env:GRAPE_V24_TARGET="1900000"
$env:GRAPE_V24_MIN_IMPORTANCE="0.92"
D:\Python311\python.exe scripts\highsource7\dryrun_yolo11n_graphlowrank_v24_remote.py
```

Direct prune from V23 p2200:

```powershell
$env:GRAPE_V24_SOURCE="D:\grape_combo\runs\yolo11n_lowrank_v23_p2200k_highsource7_region_img640_e150\weights\best_map50.pt"
$env:GRAPE_V24_TAG="from_v23_p2200"
$env:GRAPE_V24_TARGET="1990000"
$env:GRAPE_V24_MIN_IMPORTANCE="0.95"
D:\Python311\python.exe scripts\highsource7\dryrun_yolo11n_graphlowrank_v24_remote.py
```

## Next Gate

Do not run a 150-epoch V24 recovery unless a dry-run checkpoint is already at least competitive with V23 p1990:

```text
AP50 >= 0.97450
```

The p1950 result can be used only if the paper chooses a stricter compression point over maximum AP50.
