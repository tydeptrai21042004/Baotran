CNN MISSING-ONLY / GROUPED-UNDER-6H / 11H20 AUTO-EXPORT PACK

WHAT CHANGED
============
1. The run list was rebuilt from the latest supplied manuscript.
2. Already-complete manuscript rows are NOT rerun.
3. Fast same-experiment work is consolidated when the combined chunk is expected
   to stay under about 6 hours.
4. Heavy 100-epoch / large-sweep work remains split, usually one method per cell.
5. The 11h20 wall-clock guard remains active in every training cell.
6. If the guard fires, the cell exports:
     *_11H20_CURRENT_DONE_results.zip
   containing all non-checkpoint result/config/log metadata completed so far.
7. LoRA-Conv keeps the corrected non-finite-LR policy:
   an LR that is invalid for any required seed is excluded globally for that method;
   the entire experiment is not discarded when another LR is valid.
8. Generated per-cell YAML contains ONLY the methods requested by that cell.
   This fixes the earlier risk of method_order/method-definition mismatch and prevents
   accidental reruns of completed manuscript rows.

PACKAGE
=======
Training cells: 38
Scheduled method rows: 65
Logical experiments: 9
Merge cells: 7

FAST GROUPS
===========
- U01 USPS/MobileNetV3-Small 10ep: all 3 missing methods in one cell.
- N02 Flowers17/MobileNetV3-Small 10ep: all 10 blank methods in one cell.
- N01 STL10/EfficientNet-B0 10ep: 2 grouped cells, 5 methods each.
- N04 Flowers17/ResNet-18 100ep: 2 grouped cells, 5 methods each.

HEAVY WORK KEPT SPLIT
=====================
- C13 EuroSAT/MobileNetV3-Small 25ep: only DT1D, Conv-r4, BAM are missing;
  each is kept separate because the full validation sweep can exceed 6 hours.
- C11 Oxford-IIIT Pet/EfficientNet-B0 100ep: missing non-Prompt rows are kept
  one method per cell.
- Table 03 DTD/ResNet-50 100ep: missing rows kept one method per cell.
- Table 04 Flowers102/ResNet-50 100ep: missing rows kept one method per cell.
- N03 STL10/DenseNet-121 100ep: all 10 blank methods kept one method per cell.

PROMPT/VPT NOTE
===============
The current manuscript has a blank Prompt/VPT row in C10/C11. It is NOT scheduled
here because the current repo's prompt path is not wired to the torchvision
EfficientNet-B0 --backbone path; it falls through to the custom args.model builder.
Running it as-is would therefore produce the wrong backbone/protocol. This pack
does not silently change that baseline implementation.

HOW TO RUN
==========
1. Open 00_FASTEST_TO_SLOWEST_ORDER.txt.
2. Run each file in 01_RUN_CELLS_FASTEST_TO_SLOWEST as an independent Kaggle cell/session.
3. Keep each *_DONE_ONLY_results.zip output.
4. If 11h20 is reached, also keep *_11H20_CURRENT_DONE_results.zip and the resumable
   *_results.zip state export.
5. For experiments split across multiple cells, upload the DONE_ONLY ZIPs and run
   the matching merge cell in 02_MERGE_AFTER_SPLIT_EXPERIMENTS.

See 03_POLICY_AND_NOTES for the exact manuscript scope and timeout behavior.
