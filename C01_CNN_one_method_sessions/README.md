# C01 CNN — one method per Kaggle session

Repository commit is pinned to `fda0047228b3cf048ebb43d93f3970574fbef9ef`, the exact commit recorded by the supplied successful C01 run.

## What changed

- One Kaggle session runs **one method only**. There is no loop over methods.
- Each method writes its own ZIP: `C01_<method>_results.zip`.
- The scheduler re-computes remaining time **after every GPU wave**. It no longer stops after one precomputed batch.
- Resume is method-local. Upload only that method's result ZIP to the next session and rerun the same cell.
- The DT1D script can import DT1D progress from the older `PACK_C01_CNN_results*.zip`, so the four completed DT1D tuning jobs are not rerun.
- The scientific protocol is unchanged: common LR grid × seeds 0/1/2, shared-LR selection from validation only, then independent final retraining/testing.

## Recommended order

1. `01_dt1d_KAGGLE.md`
2. `02_ssf_KAGGLE.md`
3. `03_conv_r4_KAGGLE.md`
4. `04_bam_KAGGLE.md`
5. `05_residual_KAGGLE.md`
6. `06_lora_conv_KAGGLE.md`
7. `07_sidetune_KAGGLE.md`
8. `08_full_KAGGLE.md`

A method may require more than one Kaggle session. If `TASK_STATUS.json` says `"complete": false`, attach the produced `C01_<method>_results.zip` as Kaggle input and rerun the **same** method cell. Do not move to the next method until the current method reports `"complete": true` if you want a clean sequential workflow.

## Kaggle usage

- Enable GPU accelerator (the code uses up to 2 GPUs when Kaggle provides them).
- Paste one `*_KAGGLE.md` cell into the notebook and run it.
- Download the produced `C01_<method>_results.zip`.
- For a resume, add that ZIP as a Kaggle input and rerun the same cell.

The `.sh` versions contain the same code without the `%%bash` notebook magic.
