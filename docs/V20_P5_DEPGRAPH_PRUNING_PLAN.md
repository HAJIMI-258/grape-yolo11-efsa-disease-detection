# V20: Accuracy-Preserving P5 Dependency Pruning

## Hard requirement

The final compressed model is accepted only when its AP50-selected checkpoint satisfies:

```text
AP50 >= 0.96878
```

V14, V18, and V19 do not satisfy this condition and are not final-model candidates.

## Why this line is different

All previous lightweight models were newly designed narrow networks. Even with KD, they had to relearn a different channel basis and repeatedly lost `healthy`, `brown_spot`, and `mites_disease` discrimination.

V20 starts from the trained full YOLO11n AP50 checkpoint and physically removes low-importance channels from the deep P5 route only. It does not retrain a hand-designed width-scaled student.

Protected components:

- backbone layers 0-6, including P2/P3/P4 features;
- P3 and P4 PAN outputs;
- all classification towers;
- all box-regression towers and DFL;
- detection scales, anchors, decoding, and NMS.

Pruned output projections:

- layer 7 P5 downsample;
- layer 8 P5 C3k2 output;
- layer 9 SPPF output;
- layer 10 C2PSA output;
- layer 20 deepest bottom-up downsample;
- layer 22 final P5 output.

Channels are ranked with trained filter L2 energy multiplied by paired BN scale. Torch-Pruning DepGraph removes the coupled BN channels and downstream input channels, preserving graph consistency. This is not the prefix slicing used by V17.

## Optional source-checkpoint soup

Before pruning, `build_yolo11n_baseline_soup_highsource7.py` can interpolate the AP50-selected and mAP50-95-selected full YOLO11n checkpoints. It validates each interpolation coefficient and writes `D:\grape_combo\baseline_soup\soup_results.csv`.

Use the soup as `GRAPE_PRUNE_SOURCE` only when its measured AP50 is strictly greater than `0.96878`. Otherwise prune the original AP50-selected checkpoint. The soup is optional and does not change deployment parameters.

```powershell
D:\Python311\python.exe scripts\highsource7\build_yolo11n_baseline_soup_highsource7.py
```

## Compression ladder

| Profile | Deep P5 | Deep downsample | P5 head | Expected params | Approx. reduction |
| ------- | ------: | --------------: | ------: | --------------: | ----------------: |
| `p233`  |     232 |             120 |     232 |          2.334M |               10% |
| `p226`  |     224 |             120 |     224 |          2.256M |               13% |
| `p210`  |     208 |             112 |     208 |          2.096M |               19% |
| `p196`  |     192 |             112 |     192 |          1.955M |               25% |

Run only `p233` first. A smaller profile is allowed only after the previous profile reaches or exceeds the baseline AP50.

## Recovery protocol

- source: AP50-selected full YOLO11n checkpoint, or a validated better soup;
- teacher: original mAP50-95-selected YOLO11n checkpoint;
- image size: 640;
- epochs: 150;
- batch: 64;
- seed: 42;
- original data split and augmentation;
- layers 0-6 frozen throughout recovery;
- conservative training-only KD;
- SGD recovery learning rate `0.001` with cosine decay;
- save both ordinary `best.pt` and AP50-selected `best_map50.pt`.

This is a structured-compression experiment, not a same-start architecture comparison. The low recovery LR and frozen retained backbone must be disclosed.

## Installation and dry run

```powershell
pip install -r requirements-pruning.txt

git checkout codex/v20-p5-depgraph-pruning
$env:PYTHONPATH="D:\grape_combo\grape-yolo11-efsa-disease-detection;D:\grape_combo"
$env:GRAPE_P5_PROFILE="p233"

# Set this only if the soup scan beats the baseline.
# $env:GRAPE_PRUNE_SOURCE="D:\grape_combo\baseline_soup\yolo11n_highsource7_soup_best.pt"

D:\Python311\python.exe scripts\highsource7\dryrun_yolo11n_p5prune_v20_remote.py
```

The dry run must finish a 640-pixel forward pass and print a parameter count close to the profile target before any training is launched.

## Training

Run directly from the terminal. Do not create a recurring scheduled task.

```powershell
$env:GRAPE_P5_PROFILE="p233"
D:\Python311\python.exe scripts\highsource7\train_yolo11n_p5prune_v20_highsource7_region_remote.py
```

The training script first validates the physically pruned checkpoint before recovery. This separates immediate pruning damage from recovery-training effects.

## Decision rule

- `AP50 >= 0.96878`: profile passes; optionally test the next smaller profile.
- `0.96600 <= AP50 < 0.96878`: keep the checkpoint for diagnosis, but it does not satisfy the requirement.
- `AP50 < 0.96600`: stop the pruning ladder; do not add attention, GEW, wider heads, or stronger class weights.

The paper should report the smallest profile that passes. If none passes, the honest conclusion is that the measured no-drop compression frontier lies above the tested parameter budgets.
