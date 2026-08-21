# ADDITIONAL - v10 safe Kaggle work

All generated work is isolated under `ADDITIONAL/`.

- `FILL_MISSING_EXISTING_TABLES/`: only fills missing proposal/baseline rows in comparisons already present in the manuscript.
- `NEW_SMALL_CNN_DATASETS/`: exactly four new small-dataset CNN experiments, each with one assigned backbone: N01 EfficientNet-B0, N02 MobileNetV3-Small, N03 DenseNet-121, N04 ResNet-18.
- `runtime_tools/`: pre-flight runtime planners. They estimate before full training and never use a kill timer.
- `manuscript_patch/`: revised LaTeX with four blank result-entry tables matching exactly N01-N04.

## Session rule

The planning ceiling is hard-capped at `600` minutes (10 h) from shell start. Session time starts before clone/install/download. The runner keeps a 45-minute reserve inside that 10-hour planning ceiling and applies a 1.60 safety factor to exact one-epoch calibration. Before every GPU wave it checks whether the full next scientific unit is predicted to finish inside the remaining budget. If not, it does not launch the unit and writes a resumable ZIP. No runnable task uses `timeout` or an 11h50 watchdog.

Run `RUN_ORDER.csv` top to bottom. Missing rows in existing manuscript tables remain higher priority than the new N01-N04 experiments.
