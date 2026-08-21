# v10 baseline and additional-experiment policy

## Existing manuscript comparisons

`FILL_MISSING_EXISTING_TABLES/` keeps the same scientific scope: it only fills missing methods in existing manuscript tables.

| Family | Unified core baseline set |
|---|---|
| CNN | Linear, BitFit, SSF, DT1D-Adapter, Conv-Adapter r=4, BAM, Residual Adapter, LoRA-Conv, Side-Tuning, Full fine-tuning |
| ViT classification | DT1D-Adapter, VPT, Pfeiffer Adapter, Full fine-tuning, Linear probing |
| Dense prediction | Linear/task-head-only, BitFit, SSF, DT1D-Adapter, Conv-Adapter, BAM, Residual Adapter, Full fine-tuning |

## Truly additional datasets - CNN only

There are exactly four additional experiments:

- N01: STL10 / EfficientNet-B0 / 10 epochs
- N02: Oxford Flowers17 / MobileNetV3-Small / 10 epochs
- N03: STL10 / DenseNet-121 / 100 epochs
- N04: Oxford Flowers17 / ResNet-18 / 100 epochs

DenseNet-121 adds a densely connected convolutional backbone family that was not used by the paper's existing experiments before this additional block. No additional ViT or dense dataset is introduced.

## Runtime policy

Plan before launch; do not kill. Each task measures one exact calibration epoch and computes an upper-bound estimate before starting a complete LR/seed unit. The hard 10-hour total planning ceiling starts before setup, includes a 45-minute reserve, and uses a 1.60 safety factor. If the next complete unit cannot fit, it is not started and the resumable task ZIP is carried to a fresh session.
