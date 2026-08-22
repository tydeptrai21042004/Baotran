# Runtime classification for the 6-hour policy

## Safe/likely single-session tasks
- C03 missing methods: very short (10 epochs).
- C12 missing methods: short (10 epochs).
- C02 missing methods: ~3.8–3.9 h raw proxy per method; paired only with one short task.
- C13 missing methods: ~3.7 h raw proxy per method; run alone.
- C11 Full: ~4.0 h extrapolated from the completed C10 Full-vs-Linear timing ratio; run alone.
- U01 USPS/MobileNetV3-Small/10ep: new small experiment; all methods are placed in one adaptive group, but the group stops before the next task if measured runtime cannot fit.

## Cannot be safely promised under 6 h
### C01 DTD/ResNet-50/100ep
Completed C01 method timings imply roughly ~6.0–6.6 h raw for a full 18-unit method on two GPUs, before clone/install/profile overhead.
Therefore every missing C01 method is classified **multi-session** for a strict 6 h cap.

### C11 Pet/EfficientNet-B0/100ep
The completed C11 Linear and BitFit methods imply ~5.26–5.35 h raw for 18 units on two GPUs.
That leaves too little room for setup/profiling, and DT1D is slower than Linear in C10.
So:
- C11 DT1D: **likely >6 h**.
- C11 SSF and Conv-Adapter variants: **borderline / not safe to promise <6 h**.
- C11 Full: likely <6 h and gets its own standalone session.

### N03 STL10/DenseNet-121/100ep
Measured DT1D lower bound is **13.71 h on two GPUs**, so one method definitely cannot run under 6 h (or even 12 h).
