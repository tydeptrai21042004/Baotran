# N03 single-method 12-hour warning
One measured N03 DT1D 100-epoch tuning unit took 5485.91 s (~91.43 min). Fair selection requires 15 tuning units plus 3 final units = 18. Even with ideal two-GPU packing, 18*5485.91/2 = 49,373 s = 13.71 h, before setup/profile overhead. Therefore one complete N03 DT1D method cannot finish inside one 12-hour Kaggle session; use multi-session resume only.
