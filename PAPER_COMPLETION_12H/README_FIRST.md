# PAPER_COMPLETION_12H

## Purpose
1. Phase 1 fills the CNN rows blank in the supplied current manuscript.
2. Phase 2 adds USPS + MobileNetV3-Small + 10 epochs + seeds 0,1,2.

## One script = one Kaggle bash cell/session
Paste exactly one `.sh` into one Kaggle `%%bash` cell. Follow `PHASE1_RUN_ORDER.csv`, then `PHASE2_RUN_ORDER.csv`.

## Runtime protection
The existing audited runner records shell start before setup, measures one real epoch, caps planning at 600 min (10 h), keeps 45 min reserve, uses safety factor 1.60 and max 2 GPUs, and refuses predicted over-budget waves. Progress is resumable. Static estimates are only planning aids; `PRE_FLIGHT_ESTIMATE.json` is authoritative.

## Important
- Remove Prompt/VPT blank rows from CNN EfficientNet tables instead of forcing VPT onto a CNN.
- Remove fully blank N01--N04 templates from a submission-ready manuscript unless completed.
- N03 DT1D cannot complete one full method inside one 12-hour session; see the warning file.
