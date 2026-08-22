# N03: one method cannot fit under 6 hours

The measured N03 DT1D 100-epoch tuning unit is 5485.91 s (~91.43 min).
The fair method protocol requires 18 units (15 LR/seed tuning + 3 final seeds).

Even with ideal 2-GPU packing:
18 * 5485.91 / 2 = 49,373 s = 13.71 h.

Therefore N03 DT1D is not merely over 6 h; it is over 12 h for one complete method.
It must be multi-session, or N03 should be removed from the final paper.
