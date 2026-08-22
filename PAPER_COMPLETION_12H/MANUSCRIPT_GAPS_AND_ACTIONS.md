# Current manuscript CNN gaps

## Phase 1 scripts
- DTD / ResNet-50 / 100 ep: DT1D, Full, SSF, Conv-r2, Conv-r4, Residual.
- Flowers102 / ResNet-50 / 100 ep: DT1D, Full, SSF, Conv-r2/r4/r6/r8, Residual.
- Flowers102 / ResNet-18 / 10 ep: BAM, Residual, LoRA-Conv, Side-Tuning.
- Oxford-IIIT Pet / EfficientNet-B0 / 100 ep: DT1D, Full, SSF, Conv-r2/r4/r6/r8.
- Caltech101 / ResNet-18 / 10 ep: Full, LoRA-Conv, Side-Tuning.
- EuroSAT / MobileNetV3-Small / 25 ep: DT1D, Full, Conv-r4, BAM.

## Remove rather than run
- Prompt/VPT in C10/C11 CNN tables.
- Fully blank N01--N04 authoring templates if they are not completed.

## Phase 2 new CNN result
USPS / MobileNetV3-Small / 10 ep / seeds 0,1,2: Linear, DT1D, BitFit, SSF, Conv-r4, BAM, Full. One method per safe/resumable session. The cell checks cloned-repository USPS support before real training and fails early if unavailable.
