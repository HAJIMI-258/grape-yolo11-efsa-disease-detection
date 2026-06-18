# HighSource7 YOLO11 Fair Protocol

This file locks the same-start architecture-comparison protocol for the grape disease detection experiments and separates it from teacher-assisted structured-compression experiments.

## Fixed Items

- Dataset: `D:\grape_combo\highsource7_region_yolo_v1\data.yaml`
- Split: unchanged train / val / test from `highsource7_region_yolo_v1`
- Image size: `640`
- Epochs: `150`
- Batch size: `64`
- Workers: `4`
- Device: `0`
- Seed: `42`
- AMP: `True`
- Cache: `False`
- Early stopping: disabled with `patience=0`
- Base detection loss weights: `box=8.0`, `cls=0.65`, `dfl=1.7`
- Augmentation:
    - `mosaic=0.45`
    - `mixup=0.0`
    - `copy_paste=0.0`
    - `close_mosaic=20`
    - default HSV / flip / translate / scale from the same Ultralytics config
- Validation:
    - same split
    - same `imgsz=640`
    - same NMS / confidence defaults from the training validator

## Allowed Method Changes

- Student model architecture under the same data and training protocol.
- Knowledge distillation losses used only during training.
- Teacher model strength, as long as the teacher is trained under the same dataset split and image size.
- Per-class adaptive distillation weights derived from the baseline-vs-student per-class validation gap.

## Not Allowed In Same-Start Main Table

- Increasing `imgsz`.
- Increasing `epochs`.
- Changing train / val / test split.
- Changing batch size only for the improved deployment model.
- Changing optimizer, LR schedule, warmup, or augmentation only for the improved deployment model.
- Reporting an improved model initialized from the dataset-trained baseline as if it were trained from the same start.

## Current Same-Start Reference Runs

| Run                                                        | Params | GFLOPs | Best mAP50 | Best mAP50-95 | Notes                                                                                                      |
| ---------------------------------------------------------- | -----: | -----: | ---------: | ------------: | ---------------------------------------------------------------------------------------------------------- |
| `yolo11n_highsource7_region_fast_img640_e150`              | 2.591M |    6.3 |    0.96878 |       0.81389 | Full YOLO11n baseline                                                                                      |
| `yolo11n_gapkd_full_highsource7_region_img640_e150`        | 2.591M |    6.3 |     failed |        failed | Full YOLO11n + conservative teacher KD; stopped after CUDA OOM fallback and Windows page-file error        |
| `yolo11n_width20_kd_highsource7_region_img640_e150`        | 1.218M |   4.44 |    0.94158 |       0.78245 | 1.2M student + head KD                                                                                     |
| `yolo11n_width20_kdattn_highsource7_region_img640_e150`    | 1.218M |   4.44 |    0.93833 |       0.77971 | All-map feature-attention KD; worse than head KD, not main line                                            |
| `yolo11n_width20_gapkd_highsource7_region_img640_e150`     | 1.218M |   4.44 |    0.95429 |       0.79007 | Gap-aware weak-class KD + teacher foreground response                                                      |
| `yolo11n_width20_gapkd_v2_highsource7_region_img640_e150`  | 1.218M |   4.44 |    stopped |       stopped | Residual-gap balanced KD; stopped because early AP trailed v1 clearly                                      |
| `yolo11n_width20_gapkd_v2a_highsource7_region_img640_e150` | 1.218M |   4.44 |    0.92250 |       0.76927 | Hard-negative suppression damaged weak classes; not a main line                                            |
| `yolo11n_width20_gapkd_v11_highsource7_region_img640_e150` | 1.218M |   4.44 |    stopped |       stopped | Positive-only class KD hurt convergence after epoch 30; not a main line                                    |
| `yolo11n_width20_gapkd_v12_highsource7_region_img640_e150` | 1.218M |   4.44 |    0.95154 |       0.79536 | v1 head KD plus P3/P4 foreground feature KD; improves AP50-95 but AP50 trails v1                           |
| `yolo11n_width20_gapkd_v13_highsource7_region_img640_e150` | 1.218M |   4.44 |    0.94982 |       0.78666 | v1.2 with weaker full-run feature KD weight 0.02; does not beat v1                                         |
| `yolo11n_width20_gapkd_v14_highsource7_region_img640_e150` | 1.218M |   4.44 |    0.95802 |       0.78727 | v1 head KD plus late-ramped P3/P4 foreground feature KD from epoch 80; current same-start AP50 leader      |
| `yolo11n_width20_gapkd_v15_highsource7_region_img640_e150` | 1.218M |   4.44 |    running |       running | v14-style method with ultra-late feature KD from epoch 95                                                   |
| `yolo11n_width20_gapkd_v16_ap50teacher_img640_e150`        | 1.218M |   4.44 |    0.94957 |       0.78959 | AP50-best YOLO11n teacher was worse than the original mAP50-95-best teacher; line rejected                 |
| `yolo11n_width20_gew_gapkd_highsource7_region_img640_e150` | 1.087M |    4.2 |    0.93874 |       0.77381 | GEW migration; rejected because AP50 trails v14 and weak classes degrade                                   |

## Teacher-Assisted Structured-Compression Track

This track is separate from the same-start table. It may initialize the 1.218M student from the trained YOLO11n baseline by width-aware channel inheritance, then fine-tune it with a stronger teacher. The final deployment model remains 1.218M / 4.44 GFLOPs, but the training procedure uses a pre-existing full model and must be disclosed as compression rather than a same-start architecture comparison.

| Run | Deployment Params | Deployment GFLOPs | Status | Method |
| --- | ---: | ---: | --- | --- |
| `yolo11n_width20_gapkd_v17_strongteacher_sliminit_highsource7_region_img640_e150` | 1.218M | 4.44 | pending | Width-aware inheritance from trained YOLO11n + YOLO11s teacher + v14 late foreground GapKD |

V17 run order:

1. Train `yolo11s_highsource7_teacher_img640_e150` on the unchanged split and `imgsz=640`. It uses `batch=32, nbs=64` only to fit the teacher; the effective batch is 64 and the deployment student remains `batch=64`.
2. Do not run v17 unless the YOLO11s teacher exceeds the YOLO11n baseline AP50 or provides clearly better weak-class AP.
3. Initialize the width-0.20 student from the trained YOLO11n baseline using `slim_weight_inherit.py`.
4. Train the student for the fixed 150 epochs with the original v14 head KD and late P3/P4 foreground feature KD.
5. Report both `best.pt` and `best_map50.pt`; select the latter for the AP50-only study.

## Next Decisions

1. Keep v14 as the current same-start 1.2M leader.
2. Stop the AP50-teacher line; v16 disproved the hypothesis that the AP50-best teacher checkpoint is better for the student.
3. Do not continue hard-negative suppression, GEW slim-neck migration, or full-run feature KD.
4. Run the strong teacher first. If it does not exceed baseline AP50, do not spend a student run on v17.
5. If v17 exceeds the baseline, present it as teacher-assisted structured compression and retain v14 as the strict same-start ablation.

## Why The First GEW Migration Failed

The GEW-YOLO paper result is dataset- and protocol-specific rather than a transferable `1.2M -> 99.1` guarantee:

- On SeaShips, the paper's own YOLOv8 baseline is already `97.1` AP50; GEW-YOLO raises it to `99.1`, a 2.0-point gain on a highly saturated ship dataset.
- The paper trains for 400 epochs with batch 8, SGD, and a separate augmentation recipe, whereas HighSource7 is fixed at 150 epochs and batch 64.
- The paper diagram retains a YOLOv8n-style backbone and includes a DySnakeConv stage; the first grape migration instead narrowed the entire YOLO11 model to width 0.20 and omitted DySnakeConv.
- The first migration used a conservative residual ESSE approximation and only an 0.08 blend of WIoU, not a faithful replacement of the paper modules.
- The migrated model is 1.087M, below the paper's reported 1.2M, so it removes additional capacity from the weak fine-grained classes.

The paper remains useful as architectural motivation, but its numerical result should not be used as the expected outcome for grape disease recognition.
