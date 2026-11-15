# Go2 standing push test

The robot stands still mid-beam (x = 2 m) and gets one 0.1 s half-sine lateral push on the
trunk at t = 2 s. It survives if it is still up at t = 10 s. Each impulse is tested 512 times,
half pushed left and half right. Policies are the push-trained walking policies (B_*_p20)
fine-tuned on the stand task (`run_stand_push_experiment.sh`), one seed per condition.
Peak force = 15.45 × impulse. No trial fell before the push.

## J50: push survived 50% of the time (N·s, 95% CI; peak force in N)

| Beam width | Active tail | Locked tail | No tail |
|---|---|---|---|
| 0.20 m | **14.0** [11.6, 15.5]; 216 N | 13.6 [13.4, 13.9]; 211 N | 9.5 [9.4, 9.6]; 147 N |
| 0.10 m | **6.5** [6.3, 6.8]; 100 N | 5.1 [5.1, 5.2]; 79 N | 3.2 [2.8, 3.6]; 49 N |
| 0.06 m | **5.0** [4.9, 5.0]; 77 N | 3.8 [3.7, 3.8]; 58 N | 2.6 [2.6, 2.7]; 40 N |

Walking J50s for comparison (`results/tail_push_go2`):

| Beam width | Active | Locked | No tail |
|---|---|---|---|
| 0.20 m | 11.3 | 10.7 | 7.6 |
| 0.10 m | 6.3 | 4.9 | 3.1 |
| 0.06 m | 4.9 | 4.1 | 3.2 |

## Push direction matters

Survival (%) for left / right pushes:

| Condition, width | 10 N·s | 13 N·s | 16 N·s | 17 N·s | 18 N·s |
|---|---|---|---|---|---|
| Active, 0.20 m | 48 / 100 | 0 / 100 | 0 / 80 | 0 / 50 | 0 / 13 |
| Locked, 0.20 m | 100 / 100 | 97 / 27 | 7 / 0 | 1 / 0 | 0 / 0 |

- The active-tail policy at 0.20 m survives rightward pushes up to J50 ≈ 17 N·s (263 N peak), but leftward pushes only up to ≈ 10 N·s.
- Its two-sided survival curve sits near 50% from 11.5 to 15.5 N·s. That is why its J50 confidence interval is wide.
- The hardest push any trial survived was 19 N·s (active, 0.20 m, pushed right): 1 of 256 trials.
