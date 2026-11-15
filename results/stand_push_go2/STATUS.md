# Go2 standing push test — STATUS

Launched 2026-11-15 10:29 from e4ea414 (both GPUs idle beforehand). Launcher: run_stand_push_experiment.sh.
Manifest col 2 = SURVIVAL (1 - term_fall).

| cond | GPU | launcher PID |
|---|---|---|
| active | 0 | 94660 |
| locked | 1 | 94659 |
| notail | 0 | 94658 |

(PIDs verified from /proc environ.)

Checkpoint gate: after S_w020_p08, if term_fall ~1.0 for all three, stop and report to coordinator.

## Log
- Launched all three; all started S_w020_p08 (init = walking B_w020_p20).
- S_w020_p08 locked done: SURVIVAL 0.96094. term_fall trend (eval lines): 1.00, 1.00, 0.98, 0.38, 0.17, 0.10, 0.05, 0.09, 0.03, 0.04; reward/alive 3030 -> 762 -> 2723 -> 8183 -> ... -> 9825. Early dip then recovery.
- active / notail still in S_w020_p08 (latest term_fall 0.37 / 0.57, already recovering, not stuck at 1.0).
- S_w020_p08 done for all three. Manifest SURVIVAL: active 0.63281, locked 0.96094, notail 0.46875.
  term_fall trend (10 eval lines): active 1.00,1.00,0.96,0.90,0.78,0.78,0.82,0.58,0.47,0.37; notail 1.00,1.00,1.00,0.92,0.84,0.73,0.70,0.57,0.57,0.53. Gate passed (not stuck at 1.0); continuing to S_w020_p20.
- S_w020_p20 done: active 0.01562; locked 0.00000; notail 0.00000; (SURVIVAL). Next: eval B_w020 stand, then S_w010 stages.
- Stand eval S_w020 (0.20 m beam) done: J50 active 14.0, locked 13.625, notail 9.54 N*s. Survival at 0 N*s = 1.0 for all, so the low training-time survival of S_w020_p20 reflected hard pushes (mean ~10 N*s), not a broken policy. Active/locked now on S_w010 stages (locked/notail S_w010_p08 survival 0.047 / 0.000).
- S_w010_p08 done: SURVIVAL active 0.10156, locked 0.04688, notail 0.00000. Now in S_w010_p20.
- S_w010_p20 done: SURVIVAL locked 0.00000, notail 0.00000 (active still running). Next: eval S_w010, then S_w006.
- S_w010_p20 done for active too: SURVIVAL 0.00781.
- Stand eval S_w010 (0.10 m) done: active J50 6.48 (surv@0=1.00, @2=0.998046875, @5=0.75390625); locked J50 5.14 (surv@0=1.00, @2=1.0, @5=0.5859375); notail J50 3.15 (surv@0=1.00, @2=0.830078125, @5=0.01171875)
- S_w006_p08 done: SURVIVAL locked 0.02344, notail 0.00000.
- S_w006_p08 active: SURVIVAL 0.05469. All three now in S_w006_p20.
- Stand eval S_w006 (0.06 m) notail done: notail J50 2.61 (surv@0=1.00, @2=0.94). Notail launcher (94658) exited cleanly; all stages finished.
- Stand eval S_w006 locked done: locked J50 3.78 (surv@0=1.00, @2=1.00). Locked launcher (94659) exited cleanly.

## Final stand J50 (N*s)
| eval | active | locked | notail |
|---|---|---|---|
| S_w020 | 14.00 | 13.62 | 9.54 |
| S_w010 | 6.48 | 5.14 | 3.15 |
| S_w006 | 4.96 | 3.78 | 2.61 |

All three launchers exited cleanly; no failed stages.
