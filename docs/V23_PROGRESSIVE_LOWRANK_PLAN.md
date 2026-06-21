# V23: Progressive Sensitivity-Weighted Low-Rank Compression

## Objective

The final candidate must satisfy both:

```text
parameters < 2.0M
AP50 >= 0.96878
```

The target is to retain the positive AP50 margin established by V20/V22 while obtaining a meaningful reduction in parameters and GFLOPs.

## Starting point

V23 starts from V22 h48:

- raw parameters: `2,406,813`;
- fused parameters: `2,399,381`;
- AP50: `0.97115`;
- all three detection scales and 8400 anchors retained.

V22 proved that a heavily perturbed internal channel subspace can recover under mild p251-teacher distillation. V23 changes the compression operator rather than repeating deeper channel deletion.

## Measured V23 p2300 result

The viable first-stage budget is currently the 2.30M target with a `0.95` retained singular-energy floor:

```text
run: D:\grape_combo\runs\yolo11n_lowrank_v23_p2300k_highsource7_region_img640_e150
checkpoint: weights\best_map50.pt
```

Explicit validation of the AP50-selected checkpoint gives:

| Metric | Value |
| --- | ---: |
| raw parameters | `2,298,717` |
| fused validation parameters | `2,292,837` |
| GFLOPs | `6.17` raw / `6.1` fused |
| Precision | `0.94272` |
| Recall | `0.94243` |
| AP50 | `0.97154` |
| AP50-95 at the AP50 checkpoint | `0.81055` |
| independent best AP50-95 in `results.csv` | `0.82432` |

Per-class AP50 for the AP50-selected checkpoint:

| Class | AP50 | AP50-95 |
| --- | ---: | ---: |
| `black_rot` | `0.97642` | `0.92682` |
| `esca_black_measles` | `0.99500` | `0.99339` |
| `healthy` | `0.96028` | `0.86170` |
| `leaf_blight` | `0.99500` | `0.99482` |
| `brown_spot` | `0.95322` | `0.69009` |
| `downy_mildew` | `0.96447` | `0.65530` |
| `mites_disease` | `0.95637` | `0.55172` |

Compared with the full YOLO11n baseline, V23 p2300 improves AP50 from `0.96878` to `0.97154` while reducing parameters from `2.591M` to `2.299M` raw (`2.293M` fused). Compared with V22 h48, it slightly improves AP50 from `0.97115` to `0.97154` and removes another roughly `0.10M` parameters.

This is a successful low-rank bridge result, but it is not the final sub-2M lightweight model. Further compression should start from this checkpoint and only proceed when the dry-run AP50 remains close to the baseline.

## Method

Selected standard convolutions are replaced by a two-stage factorization:

```text
k x k Conv(c1 -> rank) + 1 x 1 Conv(rank -> c2) + original BN + original activation
```

The initialization is obtained from the singular value decomposition of the trained convolution matrix. Output channels, stride, padding, dilation, BN state, activation, feature-map dimensions, three Detect scales, and 8400-anchor ordering are unchanged.

A global planner chooses ranks in steps of eight. Each candidate rank reduction is scored by:

```text
sensitivity_weight * singular_energy_loss / parameters_saved
```

Protected or strongly penalized components:

- backbone layers 0-4 are excluded;
- P3 PAN output layer 16 is excluded;
- P3 and P4 classification towers are excluded;
- P3 regression is strongly penalized;
- deep P5/context layers receive the lowest penalty.

The rank plan is written to CSV with per-module rank, parameters saved, and retained singular-value energy.

## Why progressive compression

A one-shot move from 2.40M to below 2.0M is possible, but a two-stage sequence has a better chance of preserving the learned function:

1. **Bridge stage:** target `2,100,000` parameters.
2. **Final stage:** use the bridge `best_map50.pt` as source and target `1,950,000` parameters.

Both stages use the same p251 teacher. Existing `LowRankConv` blocks can be decomposed again at a lower rank, so the second stage continues the first rather than rebuilding from V22.

## Fixed recovery protocol

- dataset split: unchanged HighSource7;
- image size: 640;
- epochs: 150 per reported compression stage;
- batch: 64;
- seed: 42;
- base loss: `box=8.0`, `cls=0.65`, `dfl=1.7`;
- original augmentation settings;
- teacher: V20 p251 `best_map50.pt`;
- mild retention KD: `cls=0.20`, `response=0.06`, `dfl=0.02`;
- SGD, `lr0=0.0008`, cosine decay;
- layers 0-4 frozen;
- AP50-selected checkpoint saved as `best_map50.pt`.

This is a structured-compression/factorization track, not a same-start architecture comparison.

## Stage A dry run

Run directly from a terminal. Do not create a recurring scheduled task.

```powershell
git checkout codex/v23-progressive-lowrank
$env:PYTHONPATH="D:\grape_combo\grape-yolo11-efsa-disease-detection;D:\grape_combo\grape-yolo11-efsa-disease-detection\scripts\highsource7;D:\grape_combo"
$env:GRAPE_V23_TARGET="2100000"

D:\Python311\python.exe scripts\highsource7\dryrun_yolo11n_lowrank_v23_remote.py
```

Inspect:

- actual parameter count;
- GFLOPs;
- number of factorized modules;
- minimum retained singular energy;
- immediate AP50 and mAP50-95;
- generated rank-plan CSV.

## Stage A recovery

```powershell
$env:GRAPE_V23_TARGET="2100000"
D:\Python311\python.exe scripts\highsource7\train_yolo11n_lowrank_v23_highsource7_region_remote.py
```

Proceed to Stage B only when Stage A reaches at least the original baseline AP50. A preferred gate is `AP50 >= 0.97115`, which preserves the V22 result.

## Stage B dry run and recovery

Set the source to Stage A's AP50-selected checkpoint:

```powershell
$env:GRAPE_V23_SOURCE="D:\grape_combo\runs\yolo11n_lowrank_v23_p2100k_highsource7_region_img640_e150\weights\best_map50.pt"
$env:GRAPE_V23_TARGET="1950000"

D:\Python311\python.exe scripts\highsource7\dryrun_yolo11n_lowrank_v23_remote.py
D:\Python311\python.exe scripts\highsource7\train_yolo11n_lowrank_v23_highsource7_region_remote.py
```

## Decision rule

- `params < 2.0M` and `AP50 > 0.97115`: preferred final result; both compression and AP50 improve over V22.
- `params < 2.0M` and `0.96878 <= AP50 <= 0.97115`: satisfies no-drop relative to the original baseline but not the stronger V22 target.
- `AP50 < 0.96878`: reject the final stage.

Do not add GEW modules, hard negatives, strong class weights, or a redesigned head to rescue a failed low-rank stage. Diagnose the rank-plan CSV and increase the global budget instead.
