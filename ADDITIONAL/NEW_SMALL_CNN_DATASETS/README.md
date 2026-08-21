# New small-dataset CNN plan (v10)

This folder contains exactly **four** additional CNN experiments, not twelve dataset/backbone combinations.
There are no new ViT datasets and no new dense-prediction datasets.

| ID | Dataset | Backbone | Epochs | Why this backbone |
|---|---|---|---:|---|
| N01 | STL10 | EfficientNet-B0 | 10 | compact compound-scaled CNN |
| N02 | Oxford Flowers17 | MobileNetV3-Small | 10 | mobile/edge-oriented CNN |
| N03 | STL10 | DenseNet-121 | 100 | modern CNN family not used as an experimental backbone elsewhere in the manuscript |
| N04 | Oxford Flowers17 | ResNet-18 | 100 | residual CNN reference |

All four use the same ten-method CNN comparison family and seeds `0,1,2`.
The common five-point learning-rate grid is selected using validation data only; one shared LR is chosen per method across the three tuning seeds before final three-seed testing.

## Kaggle session safety - plan first, never kill

Every shell records wall-clock session start before clone/install/download. The exact dataset/backbone/method is calibrated with one real epoch. Before any complete LR/seed training unit starts, the runner computes a conservative upper bound using a 1.60 safety factor and checks the remaining time against a **600-minute total planning ceiling with a 45-minute reserve**. The remaining budget is checked again before every GPU wave.

If a complete scientific unit is predicted not to fit, it is **not started**. The script writes the resumable progress ZIP and continues in a fresh Kaggle session. There is no `timeout`, 11h50 watchdog, SIGTERM timer, or deliberate kill of a training process. This keeps planned work comfortably below Kaggle's 12-hour session limit while preserving the scientific unit and shared-LR protocol.
