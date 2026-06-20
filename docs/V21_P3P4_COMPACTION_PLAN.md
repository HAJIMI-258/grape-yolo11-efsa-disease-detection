# V21: Sub-2M P3/P4 Compaction from the Passing p251 Model

## Objective

Produce a model below `2.0M` parameters while retaining or exceeding the fixed baseline:

```text
AP50 >= 0.96878
```

V21 starts from the successful V20 p251 checkpoint (`AP50=0.97343`) rather than designing another narrow network from scratch.

## Core idea

The p251 model already proves that conservative structured compression can improve AP50. Most HighSource7 failures have been caused by losing fine-grained P3/P4 representation. V21 therefore preserves:

- every backbone layer, including the complete deep context path;
- the top-down P4 and P3 neck;
- the trained P3 and P4 box towers;
- the trained P3 and P4 classification towers;
- DFL, decoding, anchors, and NMS.

It removes only:

- the deepest bottom-up P5 PAN path (layers 20-22);
- the third P5 box/classification tower.

The p251 checkpoint's first two detection towers are copied exactly. The compact model predicts from strides 8 and 16, yielding `6400 + 1600 = 8000` anchors at `imgsz=640`.

## Why this can reach sub-2M

The removed P5 PAN block and P5 detection tower contain substantially more parameters than the shallow P3/P4 heads. The surgery has a hard parameter guard and aborts unless the resulting graph is below `2,000,000` parameters.

Unlike V18 and V19, V21 does not redistribute channels or initialize new feature extractors. It deletes a complete scale-specific path from a model that already passes the AP50 requirement.

## Training-only recovery

If the immediate compact checkpoint is below the target, recovery training adapts only the retained P3/P4 neck and head:

- source and teacher: V20 p251 `best_map50.pt`;
- fixed dataset split, `imgsz=640`, `epochs=150`, `batch=64`, `seed=42`;
- backbone layers 0-10 frozen;
- SGD with `lr0=0.001` and cosine decay;
- conservative P3/P4-only teacher retention KD;
- no class reweighting, hard negatives, GEW modules, WIoU, or new attention;
- save `best_map50.pt`.

This is a structured-compression experiment and must be reported as such.

## Dry run first

Run directly from the terminal; do not create a recurring scheduled task.

```powershell
git checkout codex/v21-p3p4-compact
$env:PYTHONPATH="D:\grape_combo\grape-yolo11-efsa-disease-detection;D:\grape_combo"

D:\Python311\python.exe scripts\highsource7\dryrun_yolo11n_p3p4compact_v21_remote.py
```

The dry run reports:

- exact raw and fused parameter counts;
- GFLOPs;
- immediate Precision, Recall, AP50, and mAP50-95;
- the saved pre-recovery checkpoint.

## Training

```powershell
D:\Python311\python.exe scripts\highsource7\train_yolo11n_p3p4compact_v21_highsource7_region_remote.py
```

## Decision rule

- Final `AP50 >= 0.96878` and parameters `<2.0M`: success and new main candidate.
- Final `0.96500 <= AP50 < 0.96878`: near miss but does not satisfy the requirement.
- Final `AP50 < 0.96500`: archive the two-scale line.

The immediate dry-run AP50 is also important. If scale deletion causes a large drop before training, the failure is architectural rather than an optimizer issue.
