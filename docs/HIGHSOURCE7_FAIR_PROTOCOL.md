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
| `yolo11n_width20_gapkd_v14_highsource7_region_img640_e150` | 1.218M |   4.44 |    0.95802 |       0.78727 | v1 head KD plus late-ramped P3/P4 foreground feature KD from epoch 80; current AP50-leading 1.2M run       |
| `yolo11n_width20_gapkd_v15_highsource7_region_img640_e150` | 1.218M |   4.44 |    running |       running | v14-style method with ultra-late feature KD from epoch 95 to further protect classification convergence    |
| `yolo11n_width20_gew_gapkd_highsource7_region_img640_e150` | 1.087M |    4.2 |    0.93874 |       0.77381 | GEW-YOLO migration; completed but rejected because AP50 trails v14 by 1.93 points and weak classes degrade |

## Next Decisions

1. Keep `yolo11n_width20_gapkd_v14_highsource7_region_img640_e150` as the current AP50-leading 1.2M improved run.
2. Do not continue `hard negative + strong class weights`; v2A lowered AP50 by more than 3 points.
3. Keep `yolo11n_width20_gapkd_v12_highsource7_region_img640_e150` as the AP50-95-leading 1.2M auxiliary run.
4. Continue only conservative variants of v1/v14: positive foreground feature distillation, no hard-negative suppression.
5. Run v15 because v14 is within roughly 0.08 AP50 points of the "less than 1 point AP50 drop" target versus the full YOLO11n baseline.
6. Do not continue the GEW slim-neck migration for this dataset; its capacity distribution hurts `healthy`, `brown_spot`, and `mites_disease`.

## GEW-YOLO Paper Migration

The 2025 Scientific Reports ship-detection paper `41598_2025_Article_21887.pdf` reports a GEW-YOLO variant based on YOLOv8n with:

- GSConvns and VoVGSCSPns for a lightweight slim neck.
- ESSE semantic-spatial enhancement attention.
- Wise-IoU for box regression.

The paper reports `1.2M` parameters and `99.1` mAP50 on SeaShips, but this is not directly transferable as a result claim because SeaShips is easier than fine-grained grape disease spot detection. The transferable design is the slim-neck structure and ESSE attention. The first grape migration kept the fixed HighSource7 protocol and tested the modules as `yolo11n_width20_gew_gapkd_highsource7_region_img640_e150`.

The migrated GEW run completed at 150 epochs with best AP50 `0.93874` at epoch 120 and best AP50-95 `0.77381` at epoch 142. Final validation AP50 was `0.93471`. Per-class AP50 for the weak classes was `healthy=0.911`, `brown_spot=0.893`, and `mites_disease=0.828`, so the GEW slim neck is not retained as a main-line method for HighSource7.
