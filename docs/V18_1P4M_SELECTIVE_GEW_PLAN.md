# V18: 1.4M Capacity Reallocation and Selective GEW

## Objective

Reach or exceed the fixed YOLO11n baseline `AP50=0.96878` while keeping the deployment model near `1.4M` parameters. All main comparisons retain the locked HighSource7 protocol: same split, `imgsz=640`, `epochs=150`, `batch=64`, and `seed=42`.

## Why the GEW paper is split instead of copied wholesale

The GEW-YOLO ablations are strongly dataset-dependent:

- SeaShips: baseline `97.1`; GSConvns `97.6`; ESSE `98.3`; Wise-IoU `98.4`; full model `99.1` AP50.
- Dockship: baseline `78.3`; each isolated module can reduce AP50, while the full combination reaches `82.1`.
- Infrared Offshore Ship: baseline `89.4`; GSConvns `90.5`; ESSE `90.2`; Wise-IoU `89.4`; full model `91.7`.

Therefore the grape experiment does not repeat the rejected all-neck replacement. It uses GSConvns only as a parameter exchange mechanism and tests ESSE without changing the regression loss. Wise-IoU is postponed until the architecture result is known.

## Shared design principles

- Preserve YOLO11n-compatible channels through backbone layer 6 (`16/32/64/128/128`) so exact-shape COCO tensors transfer into the shallow and P4 feature extractor.
- Keep the deep P5 backbone compact at 128 channels.
- Allocate most additional capacity to P3/P4, where color, lesion boundaries, and texture are retained.
- Reuse the proven v14 KD schedule and the original mAP50-95-selected YOLO11n teacher.
- Save `best_map50.pt` in addition to the ordinary Ultralytics checkpoint.

## V18A: parameter-matched capacity control

Config: `configs/models/yolo11n_1p4m_realloc_highsource7_region.yaml`

- Standard YOLO11 operators only.
- P3/P4/P5 detection features: `72/112/112`.
- Expected deployment budget: approximately `1.4M` parameters.

Purpose: determine whether the missing AP is mainly caused by the 1.2M capacity ceiling.

## V18B: selective GEW at the same budget

Config: `configs/models/yolo11n_1p4m_selective_gew_highsource7_region.yaml`

- Same pretrained-compatible backbone as V18A.
- Retains every C3k2 fusion block.
- Replaces only the deepest bottom-up downsample with one GSConvns block.
- Uses the saved parameters for a full 128-channel P5 head and ESSE on P3/P4.
- P3/P4/P5 detection features: `72/112/128`.
- ESSE residual gate: initial `0.04`, bounded by `0.20`.
- No Wise-IoU in this run.

Purpose: test whether the paper's lightweight fusion can fund lesion-sensitive feature enhancement without repeating the capacity collapse of the 1.087M GEW experiment.

## Run order

```powershell
$env:PYTHONPATH="D:\grape_combo\grape-yolo11-efsa-disease-detection;D:\grape_combo"

D:\Python311\python.exe scripts\highsource7\train_yolo11n_1p4m_gapkd_v18a_realloc_highsource7_region_remote.py
D:\Python311\python.exe scripts\highsource7\train_yolo11n_1p4m_gapkd_v18b_selective_gew_highsource7_region_remote.py
```

The trainer loads exact-shape COCO weights from `D:\grape_mypfe6\yolo11n.pt`, then trains from epoch 0 under the fixed protocol. It does not inherit the dataset-trained baseline.

## Decision rule

- Success: `best_map50.pt AP50 >= 0.96878`.
- Near miss: `0.96500-0.96877`; only then consider a separate Wise-IoU ablation.
- Failure: `<0.96500`; do not add more KD variants. Diagnose per-class AP and capacity allocation first.

Both V18A and V18B must complete before attributing a gain to GEW modules. If V18A wins, the result is capacity reallocation rather than GEW. If V18B wins at a similar parameter count, GSConvns-funded ESSE is supported by the ablation.

## Completed Results

Both v18 runs failed the decision rule and are archived as negative ablations. They should not replace the v14 1.2M main line.

| Variant                    |       Params | GFLOPs | Best AP50 | Best mAP50-95 | Outcome                                                     |
| -------------------------- | -----------: | -----: | --------: | ------------: | ----------------------------------------------------------- |
| V18A capacity reallocation | 1.406M fused |    5.4 | `0.95066` |     `0.79339` | Worse than v14; capacity alone did not recover weak classes |
| V18B selective GEW         | 1.401M fused |    5.5 | `0.95609` |     `0.77590` | Better than V18A, still below v14 and far below baseline    |

V18A completed a full 150-epoch run. Its AP50-selected checkpoint reached `0.95066` at epoch 106, and its mAP50-95-selected checkpoint reached `0.79339` at epoch 147.

V18B also completed before a scheduler mistake relaunched the task at `23:59` and overwrote `results.csv`. The original checkpoints survived, so the metrics above were recomputed by validating:

- `weights/best_map50.pt`: `P=0.93688`, `R=0.88153`, `AP50=0.95609`, `mAP50-95=0.77122`;
- `weights/best.pt`: `P=0.92868`, `R=0.88385`, `AP50=0.95296`, `mAP50-95=0.77590`.

V18B per-class AP50 from `best_map50.pt`:

| Class              |      AP50 |  mAP50-95 |
| ------------------ | --------: | --------: |
| black_rot          | `0.98119` | `0.89337` |
| esca_black_measles | `0.99500` | `0.96029` |
| healthy            | `0.93308` | `0.79311` |
| leaf_blight        | `0.99500` | `0.94470` |
| brown_spot         | `0.88837` | `0.62780` |
| downy_mildew       | `0.95658` | `0.64476` |
| mites_disease      | `0.94340` | `0.53451` |

The result does not support continuing the 1.4M GEW branch. The weak fine-grained classes remain the bottleneck, especially `brown_spot` and localization quality for `mites_disease`. The current best deployable student remains `yolo11n_width20_gapkd_v14_highsource7_region_img640_e150`.
