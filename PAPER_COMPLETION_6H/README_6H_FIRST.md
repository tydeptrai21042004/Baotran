# PAPER_COMPLETION_6H

This version groups Kaggle work into sessions with a strict **<6 hour target**.

## How the hard session budget works

Every grouped session:
- starts one shared outer clock;
- passes that same clock to every child method;
- sets `PACK_SAFE_MINUTES=345` (5 h 45 min from outer session start);
- keeps an additional 20-min task reserve;
- uses a 1.25 runtime safety factor;
- calibrates one real epoch before planning work;
- refuses a GPU wave predicted not to fit;
- saves a resumable `*_results.zip`.

So a second/third method does **not** get a fresh 6-hour clock.

## Run order

Run `GROUPED_6H_SESSIONS/SESSION_01.sh` through `SESSION_14_NEW_U01.sh`.

For C01 and the borderline C11 methods, use:
`GROUPED_6H_SESSIONS/LONG_OR_BORDERLINE_REPEATABLE/`.

Those are intentionally one method per repeatable <=6 h session. Attach the previous task's result ZIP and rerun the same session until complete.

## Important

A static estimate is only a planning guide. The real one-epoch preflight is authoritative.
If measured runtime on the assigned Kaggle GPU is worse than the proxy, the task stops safely and writes a resumable ZIP rather than crossing the 6-hour policy.

See:
- `GROUPED_6H_SESSION_PLAN.csv`
- `RUNTIME_6H_CLASSIFICATION.md`
- `N03_OVER_6H_AND_12H_WARNING.md`
