# V17 Strong-Teacher Slim-Initialization Experiment

## Purpose

V14 is the strongest strict same-start 1.218M run (`AP50=0.95802`) but remains below the full YOLO11n baseline (`0.96878`). V17 changes the problem from a same-start architecture comparison to explicit teacher-assisted structured compression:

1. Train a stronger YOLO11s teacher on the unchanged HighSource7 split and `imgsz=640`.
2. Initialize the existing 1.218M width-0.20 student from the trained YOLO11n baseline by copying the common channel prefix throughout the topology.
3. Fine-tune the inherited student with the proven v14 head KD and late P3/P4 foreground feature KD.

The deployment architecture does not change: `1.218M` parameters and `4.44` GFLOPs. The additional models are training-only.

## Files

- `scripts/highsource7/slim_weight_inherit.py`
- `scripts/highsource7/train_yolo11s_teacher_highsource7_region_remote.py`
- `scripts/highsource7/train_yolo11n_width20_gapkd_v17_strongteacher_sliminit_highsource7_region_remote.py`
- `tests/test_slim_weight_inherit.py`

## Run Order

From the repository root on the RTX 3090 host:

```powershell
$env:PYTHONPATH="D:\grape_combo\grape-yolo11-efsa-disease-detection;D:\grape_combo"
D:\Python311\python.exe scripts\highsource7\train_yolo11s_teacher_highsource7_region_remote.py
```

The default strong-teacher initialization path is `D:\grape_mypfe6\yolo11s.pt`. Override it when necessary:

```powershell
$env:GRAPE_YOLO11S_WEIGHTS="D:\models\yolo11s.pt"
```

After the teacher completes, inspect its AP50 and weak-class AP values. Do not run the student unless the teacher exceeds the YOLO11n baseline AP50 or gives materially better `healthy`, `brown_spot`, `downy_mildew`, and `mites_disease` AP.

Then run:

```powershell
$env:GRAPE_SLIM_INIT_WEIGHTS="D:\grape_combo\runs\yolo11n_highsource7_region_fast_img640_e150\weights\best.pt"
$env:GRAPE_STRONG_TEACHER_WEIGHTS="D:\grape_combo\runs\yolo11s_highsource7_teacher_img640_e150\weights\best.pt"
$env:GRAPE_TEACHER_CHUNK="16"
D:\Python311\python.exe scripts\highsource7\train_yolo11n_width20_gapkd_v17_strongteacher_sliminit_highsource7_region_remote.py
```

If teacher inference exceeds GPU memory, lower `GRAPE_TEACHER_CHUNK` to `8`. This changes only teacher forward micro-batching and does not change the student batch of 64.

## Outputs

```text
D:\grape_combo\runs\yolo11n_width20_gapkd_v17_strongteacher_sliminit_highsource7_region_img640_e150\weights\best.pt
D:\grape_combo\runs\yolo11n_width20_gapkd_v17_strongteacher_sliminit_highsource7_region_img640_e150\weights\best_map50.pt
```

Use `best_map50.pt` for the AP50-only study.

## Reporting Rule

V17 inherits from the dataset-trained baseline. It must be described as teacher-assisted structured compression or pruning-style fine-tuning, not as a model trained from the same initialization as the baseline. Keep v14 in the paper as the strict same-start ablation.
