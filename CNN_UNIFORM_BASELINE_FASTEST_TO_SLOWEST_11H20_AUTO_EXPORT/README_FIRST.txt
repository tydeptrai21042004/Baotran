CNN UNIFORM BASELINE — FASTEST TO SLOWEST

1) Open 00_FASTEST_TO_SLOWEST_ORDER.txt.
2) Run files in 01_RUN_CELLS_FASTEST_TO_SLOWEST from 001 upward.
3) Each file is still an independent Kaggle session.
4) If a cell reaches 11h20, it exports *_11H20_CURRENT_DONE_results.zip with current completed work.
5) Resume with that state in a later session as supported by the cell.
6) Run the matching merge cell in 02_MERGE_AFTER_EXPERIMENT_FINISHES only after all chunks for that experiment are done.

The ranking is expected compute order, not a promise of exact wall-clock time.
No experimental hyperparameters were changed by the sorting pass; only file order and, for grouped cells, method execution order were changed.
