# Remaining Kaggle sessions — strict under-10-hour edition

This folder contains only the experiments still worth running for the current manuscript plan.
Every numbered file in `sessions/` runs **one method only**. There is no long+short pairing such as the old `C02 DT1D + C12 Full` session.

## Hard runtime policy

- `PACK_SAFE_MINUTES=570` = planner ceiling at 9 h 30 min from the start of the shell, including clone/install/download.
- 20 min are reserved internally before launching a new GPU wave.
- An outer GNU `timeout` kills the whole task group at **9 h 50 min**, so the notebook cannot intentionally continue beyond 10 h.
- After a stop, completed LR/seed units are zipped. Attach the resulting `*_results.zip` to the next Kaggle session and rerun the same numbered file.
- The runner has also been corrected so a genuine numerical-divergence LR (for example LoRA-Conv producing NaN) is recorded as an invalid LR candidate rather than aborting the entire LR search. Non-numerical failures still stop for inspection.

## Important current state

- **C02 Full is complete** and is intentionally not included.
- **C02 SSF is complete** and is intentionally not included.
- **U01 BAM is complete** and is intentionally not included.
- **C02 DT1D is partial.** Your supplied archive has 12/15 tuning units complete. It still needs seed 2 at LR 0.001, 0.003, 0.005, then shared-LR selection and final seeds 0/1/2. `23_C02_dt1d.sh` refuses to run from scratch; attach `C02_dt1d_results(1).zip` first.
- C11 and N01–N04 are not included: the current plan is to remove those incomplete/redundant tables instead of spending GPU time on them.

## Run order

Run the files in `sessions/` by numeric prefix. They are sorted from the fastest planning class to the slowest. Estimates are planning ranges from the repository's measurements/proxies; Kaggle load can vary. The hard 9 h 50 min stop is authoritative.

| # | Session | Experiment | Estimate |
|---:|---|---|---:|
| 1 | `01_C03_bam.sh` | Flowers102 / ResNet-18 / 10 ep | 0.3–0.7 h |
| 2 | `02_C03_residual.sh` | Flowers102 / ResNet-18 / 10 ep | 0.3–0.7 h |
| 3 | `03_C03_lora_conv.sh` | Flowers102 / ResNet-18 / 10 ep | 0.3–0.8 h |
| 4 | `04_C03_sidetune.sh` | Flowers102 / ResNet-18 / 10 ep | 0.3–0.8 h |
| 5 | `05_U01_USPS_MV3_10EP_linear.sh` | USPS / MobileNetV3-Small / 10 ep | 0.7–1.4 h |
| 6 | `06_U01_USPS_MV3_10EP_bitfit.sh` | USPS / MobileNetV3-Small / 10 ep | 0.7–1.4 h |
| 7 | `07_U01_USPS_MV3_10EP_ssf.sh` | USPS / MobileNetV3-Small / 10 ep | 0.7–1.4 h |
| 8 | `08_U01_USPS_MV3_10EP_full.sh` | USPS / MobileNetV3-Small / 10 ep | 0.8–1.6 h |
| 9 | `09_U01_USPS_MV3_10EP_dt1d.sh` | USPS / MobileNetV3-Small / 10 ep | 0.8–1.6 h |
| 10 | `10_U01_USPS_MV3_10EP_conv_r4.sh` | USPS / MobileNetV3-Small / 10 ep | 0.8–1.6 h |
| 11 | `11_C12_full.sh` | Caltech101 / ResNet-18 / 10 ep | 0.9–1.6 h |
| 12 | `12_C12_sidetune.sh` | Caltech101 / ResNet-18 / 10 ep | 0.9–1.7 h |
| 13 | `13_C12_lora_conv.sh` | Caltech101 / ResNet-18 / 10 ep; NaN LR is skipped safely | 0.9–1.8 h |
| 14 | `14_C13_full.sh` | EuroSAT / MobileNetV3-Small / 25 ep | 3.5–5.0 h |
| 15 | `15_C13_bam.sh` | EuroSAT / MobileNetV3-Small / 25 ep | 3.5–5.0 h |
| 16 | `16_C13_conv_r4.sh` | EuroSAT / MobileNetV3-Small / 25 ep | 3.5–5.2 h |
| 17 | `17_C13_dt1d.sh` | EuroSAT / MobileNetV3-Small / 25 ep | 3.7–5.5 h |
| 18 | `18_C02_conv_r8.sh` | Flowers102 / ResNet-50 / 100 ep | 3.8–5.3 h |
| 19 | `19_C02_conv_r6.sh` | Flowers102 / ResNet-50 / 100 ep | 3.8–5.3 h |
| 20 | `20_C02_conv_r4.sh` | Flowers102 / ResNet-50 / 100 ep | 3.8–5.4 h |
| 21 | `21_C02_conv_r2.sh` | Flowers102 / ResNet-50 / 100 ep | 3.9–5.5 h |
| 22 | `22_C02_residual.sh` | Flowers102 / ResNet-50 / 100 ep | 3.9–5.5 h |
| 23 | `23_C02_dt1d.sh` | Flowers102 / ResNet-50 / 100 ep; attach C02_dt1d_results(1).zip | 4.0–5.5 h remaining |
| 24 | `24_C01_ssf.sh` | DTD / ResNet-50 / 100 ep | 6.0–8.5 h |
| 25 | `25_C01_residual.sh` | DTD / ResNet-50 / 100 ep | 6.0–8.5 h |
| 26 | `26_C01_conv_r4.sh` | DTD / ResNet-50 / 100 ep | 6.0–8.7 h |
| 27 | `27_C01_conv_r2.sh` | DTD / ResNet-50 / 100 ep | 6.1–8.8 h |
| 28 | `28_C01_full.sh` | DTD / ResNet-50 / 100 ep | 6.2–9.0 h |
| 29 | `29_C01_dt1d.sh` | DTD / ResNet-50 / 100 ep | 6.3–9.2 h |

## Kaggle usage

1. Upload this folder/ZIP as a Kaggle Dataset, or copy one numbered `.sh` into a `%%bash` cell.
2. Enable **2× T4 GPU** when available.
3. For a resume task, also attach its previous `*_results.zip` as a Kaggle input.
4. Run exactly one numbered session per Kaggle notebook session.
5. Download the generated `/kaggle/working/<TASK_ID>_results.zip` after each run.

If a task reports `complete: false`, do **not** restart it from zero. Attach that ZIP to a fresh session and rerun the exact same numbered script.
