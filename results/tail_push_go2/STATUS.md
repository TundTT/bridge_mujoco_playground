# Go2 tail push experiment STATUS

Launched: 2026-11-15 03:00 (branch tail-push-experiment @ 03d2257, worktree ~/bridge_tail_push, ROBOT=go2)

| condition | launcher PID | GPU | stage |
|---|---|---|---|
| active | 2433208 | 0 | A00 |
| locked | 2433209 | 1 | A00 |
| notail | 2433210 | 0 | A00 |

(GPUs idle at launch; two runs share GPU 0.) WandB: bridge_tail_push_go2.

## Progress
- 2026-11-15 03:06:12: ~6 min in, all three past JIT and logging. First reward lines (A00, steps: reward): active 0: -11.57, 17.0M: 22.4, 34.1M: 177.8; locked 0: -9.62, 17.0M: 48.8; notail 0: -9.80 (first eval row pending).
- "wandb: Encoding video..." appears in A00 logs (the Go1 run had a benign wandb.Video traceback at the end of stages; watch for it again, ignore unless it blocks the checkpoint).

## Problems / infra fixes
- none so far
- 2026-11-15 03:15:16: active A00 done (success=0.95312, ckpt 000153354240); locked A00 done (success=0.98438, ckpt 000153354240).
- 2026-11-15 03:16:46: notail A00 done (success=0.98438, ckpt 000153354240). All three past A00.
- 2026-11-15 03:28:17: locked A01 done (success=0.94531, ckpt 000106168320, hw=0.200).
- 2026-11-15 03:29:16: active A01 done (success=0.92188, ckpt 000106168320, hw=0.200); notail A01 done (success=0.90625, ckpt 000106168320, hw=0.200).
- 2026-11-15 03:40:20: locked A02 done (success=0.96875, ckpt 000106168320, hw=0.160).
- 2026-11-15 03:42:20: active A02 done (success=0.93750, hw=0.160); notail A02 done (success=0.91406, hw=0.160).
- 2026-11-15 03:53:53: locked A03 done (success=0.98438, hw=0.120).
- 2026-11-15 03:57:23: active A03 done (success=0.98438, hw=0.120); notail A03 done (success=0.95312, hw=0.120).
- 2026-11-15 04:07:54: locked A04 done (success=0.96875, hw=0.100).
- 2026-11-15 04:11:25: active A04 done (success=0.96875, hw=0.100).
- 2026-11-15 04:11:55: notail A04 done (success=0.93750, hw=0.100).
- 2026-11-15 04:23:28: locked A05 done (success=0.78906, hw=0.080).
- 2026-11-15 04:28:30: active A05 done (success=0.93750, hw=0.080); notail A05 done (success=0.77344, hw=0.080).
- 2026-11-15 04:37:28: locked A06 done (success=0.90625, hw=0.065).
- 2026-11-15 04:43:01: active A06 done (success=0.89844, hw=0.065).
- 2026-11-15 04:43:31: notail A06 done (success=0.82031, hw=0.065).
- 2026-11-15 04:52:34: locked A07 done (success=0.34375, hw=0.050).
- 2026-11-15 04:59:36: active A07 done (success=0.35938, hw=0.050).
- 2026-11-15 05:00:06: notail A07 done (success=0.47656, hw=0.050).
- 2026-11-15 05:08:03: locked A08 done (success=0.65625, hw=0.045).
- 2026-11-15 05:17:08: active A08 done but success=0.00000 (hw=0.045; was 0.359 at A07). Training outcome, not an infra fault; schedule unchanged. Flagging to coordinator.
- 2026-11-15 05:17:38: notail A08 done (success=0.60938, hw=0.045).
- 2026-11-15 05:23:10: locked A09 done (success=0.82812, hw=0.040).
- 2026-11-15 05:35:15: active A09 done (success=0.76562, hw=0.040) - recovered from 0.0 at A08.
- 2026-11-15 05:35:46: notail A09 done (success=0.60938, hw=0.040).
- 2026-11-15 05:38:07: locked A10 done (success=0.56250, hw=0.035).
- 2026-11-15 05:53:14: active A10 done (success=0.85938, hw=0.035).
- 2026-11-15 05:53:45: locked A11 done (success=0.73438, hw=0.030; Phase A complete for locked); notail A10 done (success=0.57031, hw=0.035).
- 2026-11-15 06:05:22: locked B_w020_p08 done (success=0.87500, hw=0.100, impulse [0,8]). locked Phase B started.
- 2026-11-15 06:11:44: active A11 done (success=0.79688, hw=0.030; Phase A complete for active).
- 2026-11-15 06:11:47: notail A11 done (success=0.53125, hw=0.030; Phase A complete for notail). Phase A finished for all three; Phase B in progress.
- 2026-11-15 06:16:17: locked B_w020_p14 done (success=0.47656, hw=0.100, impulse [0,14]).
- 2026-11-15 06:23:52: notail B_w020_p08 done (success=0.60938, hw=0.100, impulse [0,8]).
- 2026-11-15 06:24:52: active B_w020_p08 done (success=0.90625, hw=0.100, impulse [0,8]).
- 2026-11-15 06:27:53: locked B_w020_p20 done (success=0.29688, hw=0.100, impulse [0,20]).
- 2026-11-15 06:35:29: notail B_w020_p14 done (success=0.27344, hw=0.100, impulse [0,14]).
- 2026-11-15 06:37:15: active B_w020_p14 done (success=0.53125, hw=0.100, impulse [0,14]).
- 2026-11-15 06:39:17: locked B_w010_p08 done (success=0.18750, hw=0.050, impulse [0,8]).
- 2026-11-15 06:47:23: notail B_w020_p20 done (success=0.17188, hw=0.100, impulse [0,20]).
- 2026-11-15 06:50:25: active B_w020_p20 done (success=0.31250, hw=0.100, impulse [0,20]).
- 2026-11-15 06:50:55: locked B_w010_p14 done (success=0.04688, hw=0.050, impulse [0,14]).
- 2026-11-15 07:00:02: notail B_w010_p08 done (success=0.14062, hw=0.050, impulse [0,8]).
- 2026-11-15 07:02:03: locked B_w010_p20 done (success=0.04688, hw=0.050, impulse [0,20]).
- 2026-11-15 07:04:05: active B_w010_p08 done (success=0.24219, hw=0.050, impulse [0,8]).
- 2026-11-15 07:11:24: notail B_w010_p14 done (success=0.09375, hw=0.050, impulse [0,14]).
- 2026-11-15 07:14:25: locked B_w006_p08 done (success=0.26562, hw=0.030, impulse [0,8]).
- 2026-11-15 07:16:58: active B_w010_p14 done (success=0.13281, hw=0.050, impulse [0,14]).
- 2026-11-15 07:23:02: notail B_w010_p20 done (success=0.02344, hw=0.050, impulse [0,20]).
- 2026-11-15 07:26:35: locked B_w006_p14 done (success=0.08594, hw=0.030, impulse [0,14]).
- 2026-11-15 07:30:08: active B_w010_p20 done (success=0.09375, hw=0.050, impulse [0,20]).
- 2026-11-15 07:35:13: notail B_w006_p08 done (success=0.14062, hw=0.030, impulse [0,8]).
- 2026-11-15 07:38:25: locked B_w006_p20 done (success=0.03906, hw=0.030, impulse [0,20]). locked Phase B complete; Phase C (evals + posture) next.
- 2026-11-15 07:43:30: active B_w006_p08 done (success=0.40625, hw=0.030, impulse [0,8]).
- 2026-11-15 07:47:34: notail B_w006_p14 done (success=0.07812, hw=0.030, impulse [0,14]).
- 2026-11-15 07:54:44: locked launcher finished cleanly: 8 eval JSONs + 4 posture JSONs, no FAILED. Waiting on active and notail.
- 2026-11-15 07:57:14: active B_w006_p14 done (success=0.17188, hw=0.030, impulse [0,14]).
- 2026-11-15 08:00:17: notail B_w006_p20 done (success=0.04688, hw=0.030, impulse [0,20]). notail Phase B complete; Phase C next.
- 2026-11-15 08:10:31: active B_w006_p20 done (success=0.11719, hw=0.030, impulse [0,20]). active Phase B complete; Phase C next.
- 2026-11-15 08:15:37: notail launcher finished cleanly: 8 eval JSONs + 4 posture JSONs, no FAILED. Waiting on active (Phase C).

## FINAL (2026-11-15 08:25:44) - all Go2 runs complete (8 eval + 4 posture JSONs per condition, no FAILED)
```
active  A04_w020                 J50=8.778503046127067
active  A07_w010                 J50=3.10101092009947
active  A11_w006                 J50=5.03932584269663
active  B_w006                   J50=4.877118644067797
active  B_w010                   J50=6.313888888888889
active  B_w020                   J50=11.325
active  X_A11_at_w010            J50=5.67948717948718
active  X_B_w006_p20_at_w010     J50=5.108
locked  A04_w020                 J50=7.7298338062424
locked  A07_w010                 J50=3.2375
locked  A11_w006                 J50=4.397200827681343
locked  B_w006                   J50=4.14390435391101
locked  B_w010                   J50=4.895833333333333
locked  B_w020                   J50=10.743589743589745
locked  X_A11_at_w010            J50=4.893129770992366
locked  X_B_w006_p20_at_w010     J50=4.3875
notail  A04_w020                 J50=6.137362637362638
notail  A07_w010                 J50=3.645104895104895
notail  A11_w006                 J50=4.254098360655738
notail  B_w006                   J50=3.168261588351243
notail  B_w010                   J50=3.075725436959104
notail  B_w020                   J50=7.578947368421052
notail  X_A11_at_w010            J50=4.66243654822335
notail  X_B_w006_p20_at_w010     J50=3.191919191919192
```
