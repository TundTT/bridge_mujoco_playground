# Tail push experiment STATUS

Launched: 2026-11-11 13:32:07 (branch tail-push-experiment @ db4932a, worktree ~/bridge_tail_push)

| condition | launcher PID | GPU | stage |
|---|---|---|---|
| active | 895287 | 0 | A00 (compiling/first eval pending) |
| locked | 895289 | 1 | A00 |
| notail | 895288 | 0 | A00 |

GPUs had ~70 GB free each, so all three launched at once (two on GPU 0).
env-steps/s: ~52k per run (coordinator-measured: 17.0M steps in ~5.5 min). ETA: full schedule ~16 h from launch (~13:30 on 2026-11-11 => ~05:30 on 2026-11-12).

## Problems / infra fixes
- `uv sync` defaulted to CPython 3.14 (no jaxlib wheel). Fix: `uv venv --python 3.12` then sync (as in CLAUDE.md). No code changes.
- Bridge tests: 2 passed; jax sees 2 CUDA devices.
- PIDs correction: 895283/4/5 were the setsid wrapper PIDs (exited immediately by design); the real launcher PIDs are active=895287, notail=895288, locked=895289. Training is running normally.

## Stage log (2026-11-11 13:48:51)
- locked A00 done (success=1.0, ckpt 000153354240); now in A01. Manifest line: A00 1.00000 .../checkpoints/000153354240 {"tail_mode":"locked","bridge_half_width":0.400,...}
- active/notail A00 still running at this time.
- Benign Traceback in every A* stage log: wandb.Video rollout-video upload fails at train_jax_ppo.py:722 AFTER the checkpoint is saved; launcher continues normally. Not touching it (no effect on results). Will not treat it as a crash.
- Observation: locked A00 (153M steps) finished ~18 min after launch, faster than the ~52k steps/s ETA implies; ETA likely conservative (the unit of "steps" may differ). Will refine ETA once more stages finish.
- 2026-11-11 13:53:26: active A00 done (success=1.0, ckpt 000153354240); notail A00 done (success=1.0, ckpt 000153354240). Same benign wandb.Video Traceback in both logs. All three now in A01+.
- 2026-11-11 14:03:55: locked A01 done (success=0.98438, ckpt 000106168320, hw=0.200). Benign wandb.Video Traceback again.
- 2026-11-11 14:09:31: active A01 done (success=0.98438, ckpt 000106168320, hw=0.200).
- 2026-11-11 14:10:01: notail A01 done (success=0.97656, ckpt 000106168320, hw=0.200).
- 2026-11-11 14:19:32: locked A02 done (success=0.96094, ckpt 000106168320, hw=0.160).
- 2026-11-11 14:27:02: active A02 done (success=1.00000, ckpt 000106168320, hw=0.160).
- 2026-11-11 14:27:33: notail A02 done (success=0.94531, ckpt 000106168320, hw=0.160).
- 2026-11-11 14:36:34: locked A03 done (success=1.00000, ckpt 000153354240, hw=0.120).
- 2026-11-11 14:48:06: active A03 done (success=0.99219, ckpt 000153354240, hw=0.120).
- 2026-11-11 14:49:06: notail A03 done (success=0.96875, ckpt 000153354240, hw=0.120).
- 2026-11-11 14:54:38: locked A04 done (success=0.97656, ckpt 000153354240, hw=0.100).
- 2026-11-11 15:10:09: active A04 done (success=0.97656, ckpt 000153354240, hw=0.100).
- 2026-11-11 15:11:10: notail A04 done (success=0.93750, ckpt 000153354240, hw=0.100).
- 2026-11-11 15:15:11: locked A05 done (success=0.93750, ckpt 000200540160, hw=0.080).
- 2026-11-11 15:30:45: active A05 done (success=0.97656, ckpt 000200540160, hw=0.080).
- 2026-11-11 15:31:45: locked A06 done (success=0.93750, ckpt 000153354240, hw=0.065).
- 2026-11-11 15:32:15: notail A05 done (success=0.89844, ckpt 000200540160, hw=0.080).
- 2026-11-11 15:47:17: locked A07 done but success=0.00000 (ckpt 000200540160, hw=0.050). Experimental result, not an infra fault; schedule unchanged. Flagging to coordinator.
- 2026-11-11 15:47:46: active A06 done (success=0.98438, ckpt 000153354240, hw=0.065).
- 2026-11-11 15:49:17: notail A06 done (success=0.84375, ckpt 000153354240, hw=0.065).
- 2026-11-11 16:03:22: locked A08 done (success=0.39062, ckpt 000200540160, hw=0.045).
- 2026-11-11 16:07:24: active A07 done (success=0.82031, ckpt 000200540160, hw=0.050).
- 2026-11-11 16:09:49: notail A07 done (success=0.46875, ckpt 000200540160, hw=0.050).
- 2026-11-11 16:19:22: locked A09 done (success=0.67188, ckpt 000200540160, hw=0.040).
- 2026-11-11 16:24:54: active A08 done (success=0.86719, ckpt 000200540160, hw=0.045).
- 2026-11-11 16:27:25: notail A08 done (success=0.58594, ckpt 000200540160, hw=0.045).
- 2026-11-11 16:35:29: locked A10 done (success=0.75781, ckpt 000200540160, hw=0.035).
- 2026-11-11 16:41:53: active A09 done (success=0.89844, ckpt 000200540160, hw=0.040).
- 2026-11-11 16:44:24: notail A09 done (success=0.77344, ckpt 000200540160, hw=0.040).
- 2026-11-11 16:51:58: locked A11 done (success=0.77344, ckpt 000200540160, hw=0.030).
- 2026-11-11 17:00:33: active A10 done (success=0.89844, ckpt 000200540160, hw=0.035).
- 2026-11-11 17:03:35: locked B_w020_p08 done (success=0.78906, ckpt 000106168320, hw=0.100, impulse [0,8]). Phase A finished for locked; Phase B started.
- 2026-11-11 17:04:05: notail A10 done (success=0.84375, ckpt 000200540160, hw=0.035).
- 2026-11-11 17:15:29: locked B_w020_p14 done (success=0.41406, ckpt 000106168320, hw=0.100, impulse [0,14]).
- 2026-11-11 17:19:02: active A11 done (success=0.92188, ckpt 000200540160, hw=0.030). Phase A finished for active.
- 2026-11-11 17:23:35: notail A11 done (success=0.89844, ckpt 000200540160, hw=0.030). Phase A finished for all three; Phase B in progress.
- 2026-11-11 17:27:37: locked B_w020_p20 done (success=0.56250, ckpt 000106168320, hw=0.100, impulse [0,20]).
- 2026-11-11 17:32:39: active B_w020_p08 done (success=0.90625, ckpt 000106168320, hw=0.100, impulse [0,8]).
- 2026-11-11 17:35:42: notail B_w020_p08 done (success=0.50000, ckpt 000106168320, hw=0.100, impulse [0,8]).
- 2026-11-11 17:39:31: locked B_w010_p08 done (success=0.03906, ckpt 000106168320, hw=0.050, impulse [0,8]).
- 2026-11-11 17:46:05: active B_w020_p14 done (success=0.52344, ckpt 000106168320, hw=0.100, impulse [0,14]).
- 2026-11-11 17:48:06: notail B_w020_p14 done (success=0.27344, ckpt 000106168320, hw=0.100, impulse [0,14]).
- 2026-11-11 17:51:39: locked B_w010_p14 done (success=0.03125, ckpt 000106168320, hw=0.050, impulse [0,14]).
- 2026-11-11 17:59:14: active B_w020_p20 done (success=0.77344, ckpt 000106168320, hw=0.100, impulse [0,20]).
- 2026-11-11 18:00:15: notail B_w020_p20 done (success=0.21875, ckpt 000106168320, hw=0.100, impulse [0,20]).
- 2026-11-11 18:03:16: locked B_w010_p20 done (success=0.03125, ckpt 000106168320, hw=0.050, impulse [0,20]).
- 2026-11-11 18:12:07: active B_w010_p08 done (success=0.53125, ckpt 000106168320, hw=0.050, impulse [0,8]).
- 2026-11-11 18:12:37: notail B_w010_p08 done (success=0.37500, ckpt 000106168320, hw=0.050, impulse [0,8]).
- 2026-11-11 18:15:40: locked B_w006_p08 done (success=0.29688, ckpt 000106168320, hw=0.030, impulse [0,8]).
- 2026-11-11 18:24:47: notail B_w010_p14 done (success=0.60156, ckpt 000106168320, hw=0.050, impulse [0,14]).
- 2026-11-11 18:25:48: active B_w010_p14 done (success=0.34375, ckpt 000106168320, hw=0.050, impulse [0,14]). Note: notail got 0.60156 at same stage; training-success at differing push ranges is not comparable to the J50 eval, to be judged from Phase C.
- 2026-11-11 18:28:19: locked B_w006_p14 done (success=0.09375, ckpt 000106168320, hw=0.030, impulse [0,14]).
- 2026-11-11 18:36:57: notail B_w010_p20 done (success=0.75781, ckpt 000106168320, hw=0.050, impulse [0,20]).
- 2026-11-11 18:38:28: active B_w010_p20 done (success=0.75000, ckpt 000106168320, hw=0.050, impulse [0,20]).
- 2026-11-11 18:41:11: locked B_w006_p20 done (success=0.10938, ckpt 000106168320, hw=0.030, impulse [0,20]). Locked Phase B complete; Phase C eval next.
- 2026-11-11 18:48:19: locked launcher finished (all Phase C evals written: 6 JSONs). Waiting on active and notail.
- 2026-11-11 18:49:48: notail B_w006_p08 done (success=0.82031, ckpt 000106168320, hw=0.030, impulse [0,8]).
- 2026-11-11 18:52:21: active B_w006_p08 done (success=0.71875, ckpt 000106168320, hw=0.030, impulse [0,8]).

## Coordinator note (2026-11-11 18:53:23)
- Coordinator pulled commit 6c84576 (adds run_tail_push_cross_eval.sh + analysis update; run_tail_push_experiment.sh unchanged) and started `CONDITION=locked CUDA_VISIBLE_DEVICES=1 ./run_tail_push_cross_eval.sh` (log: results/tail_push/locked.cross_eval.log), writing eval/X_*.json.
- Reason: locked never learned 0.10 m (A07 success 0), so its width-branch J50 there measures walking failure; X_ evals test each condition's 0.06 m policy on the wider beams.
- TODO: when active and notail launchers finish, run the same script for them (active on GPU0, notail on GPU1). Include eval/X_*.json in the final results commit.
- 2026-11-11 19:02:00: notail B_w006_p14 done (success=0.76562, ckpt 000106168320, hw=0.030, impulse [0,14]).
- 2026-11-11 19:06:03: active B_w006_p14 done (success=0.82031, ckpt 000106168320, hw=0.030, impulse [0,14]).
- 2026-11-11 19:14:48: notail B_w006_p20 done (success=0.78906, ckpt 000106168320, hw=0.030, impulse [0,20]). notail Phase B complete; evals next.
- 2026-11-11 19:19:51: active B_w006_p20 done (success=0.82812, ckpt 000106168320, hw=0.030, impulse [0,20]). active Phase B complete; evals next.
- 2026-11-11 19:20:32: PROBLEM (infra): active eval A04_w020 FAILED with "cuSolver internal error" (log active/logs/eval_A04_w020.log). GPU0 was at 97/98 GB because notail's eval process (pid 1395496) grabbed ~73 GB concurrently with active's eval. Launcher continued to A07_w010. Plan: after the active/notail launchers exit, re-run the same launcher commands (existing eval JSONs are skipped, failed ones redone), serialising active/notail evals so they don't share GPU0; then cross-evals. No code/schedule change.
- 2026-11-11 19:21:20: notail launcher finished but eval B_w006 failed (same cuSolver/GPU0-memory issue). Re-ran notail launcher (CONDITION=notail CUDA_VISIBLE_DEVICES=1; log notail.launcher2.log, bash PID 1406948); all stages skipped, only B_w006 eval re-running on GPU1. Infra re-run only, no design change.
- 2026-11-11 19:25:42: notail complete (B_w006 eval re-run OK; 6 eval JSONs). active launcher finished but A04_w020 eval had failed -> re-running (active.launcher2.log, GPU0), then active cross-eval (active.cross_eval.log).
- The locked cross-eval started by the coordinator died silently (no process, empty eval_X_A11_at_w020.log, no X_ JSONs; likely killed with its parent shell). Restarted detached with setsid on GPU1 (locked.cross_eval.log), chained with notail cross-eval (notail.cross_eval.log) on GPU1. Evals are serialised per GPU to avoid the earlier cuSolver/memory failure.

## FINAL (2026-11-11 19:34:31) — all runs, evals and cross-evals complete (10 eval JSONs per condition)
J50 (N*s, 50% survival):
```
active  A04_w020                 J50=7.700787401574804
active  A07_w010                 J50=5.1330049261083746
active  A11_w006                 J50=4.9576271186440675
active  B_w006                   J50=3.5058823529411764
active  B_w010                   J50=4.871287128712871
active  B_w020                   J50=8.150375939849624
active  X_A11_at_w010            J50=5.132420091324201
active  X_A11_at_w020            J50=0.0
active  X_B_w006_p20_at_w010     J50=3.4571428571428573
active  X_B_w006_p20_at_w020     J50=0.0
locked  A04_w020                 J50=6.876288659793815
locked  A07_w010                 J50=0.0
locked  A11_w006                 J50=4.573033707865169
locked  B_w006                   J50=3.2960893854748603
locked  B_w010                   J50=3.7518005540166204
locked  B_w020                   J50=7.461538461538462
locked  X_A11_at_w010            J50=4.839622641509434
locked  X_A11_at_w020            J50=0.0
locked  X_B_w006_p20_at_w010     J50=3.4911242603550297
locked  X_B_w006_p20_at_w020     J50=0.0
notail  A04_w020                 J50=5.290322580645161
notail  A07_w010                 J50=2.963265306122449
notail  A11_w006                 J50=1.2248803827751196
notail  B_w006                   J50=1.3128205128205128
notail  B_w010                   J50=2.485207100591716
notail  B_w020                   J50=4.5840000000000005
notail  X_A11_at_w010            J50=1.4382022471910112
notail  X_A11_at_w020            J50=0.0
notail  X_B_w006_p20_at_w010     J50=2.304635761589404
notail  X_B_w006_p20_at_w020     J50=0.0
```
Notes: all X_*_at_w020 cross-evals give J50=0.0 for every condition (0.06 m-trained policies don't cross the 0.20 m beam at all -> not informative). Headline comparisons: B_w006 active 3.51 / locked 3.30 / notail 1.31; B_w010 active 4.87 / locked 3.75 / notail 2.49; X_B_w006_p20_at_w010 active 3.46 / locked 3.49 / notail 2.30. Locked A07_w010 = 0 (never learned 0.10 m). Committing manifests, eval JSONs, STATUS.md only.
