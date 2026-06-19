# HighSource7 YOLO11 Fair Protocol

This file locks the main-table protocol for the grape disease detection experiments.

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

## Not Allowed In Main Table

- Increasing `imgsz`.
- Increasing `epochs`.
- Changing train / val / test split.
- Changing batch size only for the improved model.
- Changing optimizer, LR schedule, warmup, or augmentation only for the improved model.
- Reporting an improved model trained from the baseline best checkpoint as if it were trained from the same start.

## Current Reference Runs

The table records independent best validation metrics from each run's `results.csv`. For the AP50-oriented study, AP50-selected checkpoints must be reported consistently.

| Run                                                                               | Params | GFLOPs |     Best AP50 | Best mAP50-95 | Status and interpretation                                                     |
| --------------------------------------------------------------------------------- | -----: | -----: | ------------: | ------------: | ----------------------------------------------------------------------------- |
| `yolo11n_highsource7_region_fast_img640_e150`                                     | 2.591M |    6.3 |     `0.96878` |     `0.82287` | Full YOLO11n baseline                                                         |
| `yolo11n_gapkd_full_highsource7_region_img640_e150`                               | 2.591M |    6.3 |        failed |        failed | Full YOLO11n + conservative teacher KD; stopped after host-memory failure     |
| `yolo11n_width20_kd_highsource7_region_img640_e150`                               | 1.218M |   4.44 |     `0.94158` |     `0.78245` | 1.2M student + head KD                                                        |
| `yolo11n_width20_kdattn_highsource7_region_img640_e150`                           | 1.218M |   4.44 |     `0.93833` |     `0.77971` | All-map feature-attention KD; rejected                                        |
| `yolo11n_width20_gapkd_highsource7_region_img640_e150`                            | 1.218M |   4.44 |     `0.95429` |     `0.79007` | Gap-aware weak-class KD + teacher foreground response                         |
| `yolo11n_width20_gapkd_v2_highsource7_region_img640_e150`                         | 1.218M |   4.44 |       stopped |       stopped | Residual-gap balanced KD; early trajectory clearly below v1                   |
| `yolo11n_width20_gapkd_v2a_highsource7_region_img640_e150`                        | 1.218M |   4.44 |     `0.92250` |     `0.76927` | Hard-negative suppression damaged weak classes; rejected                      |
| `yolo11n_width20_gapkd_v11_highsource7_region_img640_e150`                        | 1.218M |   4.44 |       stopped |       stopped | Positive-only class KD hurt convergence; rejected                             |
| `yolo11n_width20_gapkd_v12_highsource7_region_img640_e150`                        | 1.218M |   4.44 |     `0.95154` |     `0.79536` | Full-run P3/P4 foreground feature KD; AP50 below v1                           |
| `yolo11n_width20_gapkd_v13_highsource7_region_img640_e150`                        | 1.218M |   4.44 |     `0.94982` |     `0.78666` | Weaker full-run feature KD; rejected                                          |
| `yolo11n_width20_gapkd_v14_highsource7_region_img640_e150`                        | 1.218M |   4.44 | **`0.95802`** |     `0.78727` | Current AP50-leading 1.2M line; late-ramped P3/P4 foreground KD from epoch 80 |
| `yolo11n_width20_gapkd_v15_highsource7_region_img640_e150`                        | 1.218M |   4.44 |        failed |        failed | Stopped at epoch 32 because of host-memory allocation failure; invalid run    |
| `yolo11n_width20_gapkd_v16_ap50teacher_highsource7_region_img640_e150`            | 1.218M |   4.44 |     `0.94957` |     `0.78959` | AP50-selected teacher reduced student AP50; rejected counterexample           |
| `yolo11n_width20_gapkd_v17_strongteacher_sliminit_highsource7_region_img640_e150` | 1.213M |    4.3 |     `0.95077` |     `0.78909` | YOLO11s teacher + width-aware baseline inheritance; rejected counterexample   |
| `yolo11n_width20_gew_gapkd_highsource7_region_img640_e150`                        | 1.087M |    4.2 |     `0.93874` |     `0.77381` | GEW migration degraded weak classes; rejected                                 |
| `yolo11n_1p4m_gapkd_v18a_realloc_highsource7_region_img640_e150`                  | 1.406M |    5.4 |     `0.95066` |     `0.79339` | 1.4M capacity reallocation did not recover AP50; rejected                     |
| `yolo11n_1p4m_gapkd_v18b_selective_gew_highsource7_region_img640_e150`            | 1.401M |    5.5 |     `0.95609` |     `0.77590` | Selective GEW improved over v18A but stayed below v14; rejected               |

## Completed Counterexample Experiments

### V16: AP50-selected teacher

V16 retained the v14 student and schedule but replaced the original mAP50-95-selected YOLO11n teacher with its AP50-selected checkpoint. The best student AP50 fell from `0.95802` to `0.94957`. This shows that a teacher checkpoint with the highest standalone AP50 does not necessarily provide the most useful localization distribution for a compressed student.

### V17: stronger teacher plus slim initialization

V17 used a YOLO11s teacher and initialized the width-0.20 student by copying compatible channel prefixes from the trained YOLO11n baseline. The run completed normally:

- best AP50: `0.95077` at epoch 137;
- best mAP50-95: `0.78909` at epoch 136;
- final AP50: `0.94844`;
- final mAP50-95: `0.78463`;
- fused parameters: `1,213,351`;
- GFLOPs: `4.3`;
- inference: `1.0 ms/img`.

Weak-class AP50 remained poor: `healthy=0.896`, `brown_spot=0.904`, and `mites_disease=0.917`. The YOLO11s teacher exceeded the YOLO11n baseline by only `0.00020` AP50, so it did not provide a materially stronger signal. Width-prefix inheritance also failed to preserve the discriminative subspace required by the weak fine-grained classes. V17 trails v14 by `0.00725` AP50 and is archived.

### V18: 1.4M capacity reallocation and selective GEW

V18 tested whether relaxing the student from 1.2M to roughly 1.4M parameters could close the AP50 gap. Both runs reused the v14 KD schedule and fixed protocol, then loaded exact-shape COCO tensors rather than inheriting the dataset-trained baseline.

- V18A used only standard YOLO11 modules with more capacity assigned to P3/P4. It reached `0.95066` AP50 and `0.79339` mAP50-95.
- V18B used one GSConvns block in the deepest bottom-up downsample and two ESSE blocks on P3/P4. Its AP50-selected checkpoint reached `0.95609` AP50 and `0.77122` mAP50-95. Its mAP50-95-selected checkpoint reached `0.95296` AP50 and `0.77590` mAP50-95.

V18B's AP50 remains `0.00193` below v14 and `0.01269` below the YOLO11n baseline. The 1.4M branch therefore does not justify replacing v14. The weak classes remain concentrated in `brown_spot`, `healthy`, and `mites_disease`; adding capacity and selective GEW modules did not solve the fine-grained class-discrimination bottleneck.

## Current Decision

1. Keep `yolo11n_width20_gapkd_v14_highsource7_region_img640_e150` as the only active 1.2M main line.
2. Treat v16, v17, v18A, and v18B as negative-result/counterexample experiments in the ablation discussion.
3. Do not continue AP50-teacher selection, stronger-teacher slim initialization, hard-negative suppression, full-run feature KD, the GEW slim-neck migration, or the 1.4M selective-GEW branch.
4. The current 1.2M line does not yet satisfy the requirement `AP50 >= 0.96878`; no completed lightweight model should be described as accuracy-preserving relative to the baseline.

## GEW-YOLO Paper Migration

The 2025 Scientific Reports GEW-YOLO paper reports `1.2M` parameters and `99.1` AP50 on SeaShips, but that number is dataset- and protocol-specific. Its SeaShips baseline is already `97.1` AP50, and the full method gains 2.0 points. The paper uses 400 epochs, batch 8, SGD, DySnakeConv, GSConvns/VoVGSCSPns, ESSE, Wise-IoU, and a separate augmentation recipe. The first HighSource7 migration was not a faithful reproduction and, more importantly, the ship-domain result does not establish an expected accuracy for fine-grained grape disease classes.
