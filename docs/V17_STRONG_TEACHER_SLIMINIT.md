# V17 Strong-Teacher Slim-Initialization Experiment

## Status

**Completed and rejected as a main-line method. Do not continue this branch.**

V17 tested whether a stronger YOLO11s teacher and width-aware inheritance from the trained YOLO11n baseline could recover the accuracy lost by the 1.2M student.

## Method

1. Train a YOLO11s teacher on the unchanged HighSource7 split and `imgsz=640`.
2. Initialize the existing width-0.20 student by copying compatible channel prefixes from the trained YOLO11n baseline.
3. Fine-tune the inherited student with the v14 head GapKD and late P3/P4 foreground feature distillation.

The deployment model remained unchanged at `1,213,351` parameters and `4.3` GFLOPs.

## Final Result

| Metric | V17 result |
| --- | ---: |
| Best AP50 | `0.95077` at epoch 137 |
| Best mAP50-95 | `0.78909` at epoch 136 |
| Final AP50 | `0.94844` |
| Final mAP50-95 | `0.78463` |
| Fused parameters | `1,213,351` |
| GFLOPs | `4.3` |
| Inference | `1.0 ms/img` |
| Exit status | code `0`, no stderr |

Weak-class AP50 at the selected validation result:

| Class | AP50 |
| --- | ---: |
| `healthy` | `0.896` |
| `brown_spot` | `0.904` |
| `mites_disease` | `0.917` |

## Comparison

| Run | AP50 | Difference from v14 |
| --- | ---: | ---: |
| v14 | `0.95802` | — |
| v17 | `0.95077` | `-0.00725` |

The YOLO11s teacher exceeded the YOLO11n baseline by only `0.00020` AP50, so it did not provide a materially stronger supervisory signal. Width-aware prefix inheritance also appears to have constrained the narrow student to a subspace that did not preserve the discriminative cues required by `healthy`, `brown_spot`, and `mites_disease`.

## Conclusion

V17 does not satisfy the target that the 1.2M model must equal or exceed the YOLO11n baseline AP50. It is retained only as a negative-result experiment showing that:

- a nominally larger teacher is not useful when its AP50 advantage is negligible;
- direct channel-prefix inheritance is not sufficient for fine-grained weak-class preservation;
- the v14 late foreground GapKD schedule remains the strongest completed 1.2M line.

Do not invest further runs in the strong-teacher plus slim-initialization direction.
