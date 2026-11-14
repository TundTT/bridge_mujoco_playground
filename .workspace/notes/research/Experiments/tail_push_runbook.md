# Tail vs no-tail push-tolerance experiment — workstation runbook

Written by the coordinating Claude session (on the Windows dev laptop) for the
"Tail trainer" Claude session on the workstation (`beast`). The user has
authorised this session to direct the training work; ask the coordinator
(not the user) when unsure.

## Goal
Measure how much an actuated roll tail raises the lateral push a Go1 can
survive while crossing a narrow box beam, versus a locked tail and no tail.
Headline metric: critical push impulse J50 (N*s, 50% survival) per condition
and beam width.

## Code (branch `tail-push-experiment`)
- `mujoco_playground/_src/locomotion/go1/bridge_tail.py` — env `Go1BridgeCrossingTail`
  (13 actions; tail = AK70-10 servo; `tail_mode` active|locked).
- `mujoco_playground/_src/locomotion/go1/bridge.py` — random pushes
  (`push_config`, impulse in N*s) + `reset_info_on_new_episode` fix. Both off by
  default for `Go1BridgeCrossing`.
- `run_tail_push_experiment.sh` — fixed-budget schedule for ONE condition
  (Phase A width curriculum 0.8 m -> 0.06 m with mild pushes; Phase B push
  curriculum at 0.20 / 0.10 / 0.06 m; Phase C eval). Resumable via
  `results/tail_push/<cond>/manifest.tsv`; `STOP_AFTER=<stage>` exits 99.
- `scripts/eval_push_tolerance.py` — push sweep eval, writes JSON with J50.

## Setup (do NOT disturb ~/bridge_mujoco_playground_git: it is on branch
`humanoid` with someone's uncommitted G1 work)
```sh
cd ~/bridge_mujoco_playground_git
git fetch origin tail-push-experiment
git worktree add ~/bridge_tail_push origin/tail-push-experiment -b tail-push-experiment
cd ~/bridge_tail_push
~/.local/bin/uv --no-config sync --all-extras          # own .venv for this worktree
source .venv/bin/activate
python -c "import jax; print(jax.devices())"         # expect 2 CUDA devices
pytest mujoco_playground/_src/locomotion/locomotion_test.py -k Bridge -q   # 2 passed
```

## Launch (GPUs are shared with other users' jobs — check `nvidia-smi` first)
Each run needs roughly 10-25 GB. If each GPU has >= 45 GB free, run all three
at once (two on one GPU); otherwise run active+locked first and queue notail.
```sh
cd ~/bridge_tail_push && mkdir -p results/tail_push
CONDITION=active CUDA_VISIBLE_DEVICES=0 setsid nohup ./run_tail_push_experiment.sh > results/tail_push/active.launcher.log 2>&1 &
CONDITION=locked CUDA_VISIBLE_DEVICES=1 setsid nohup ./run_tail_push_experiment.sh > results/tail_push/locked.launcher.log 2>&1 &
CONDITION=notail CUDA_VISIBLE_DEVICES=0 setsid nohup ./run_tail_push_experiment.sh > results/tail_push/notail.launcher.log 2>&1 &
```
WandB project: `bridge_tail_push` (groups = condition).

## Status file (the coordinator reads this over Herdr)
Keep `~/bridge_tail_push/results/tail_push/STATUS.md` current: launch time,
PIDs + GPU per condition, measured env-steps/s of the first stage, ETA,
last finished stage per condition (copy manifest lines), any problems.
Update it after launching and whenever a stage finishes or something breaks.

## Rules
- Don't change the experiment design, rewards, schedules or env physics
  without asking the coordinator. Small infra fixes (paths, OOM -> fewer
  concurrent runs, re-running a crashed stage) are fine; log them in STATUS.md.
- If a stage crashes, read `results/tail_push/<cond>/logs/<stage>.log`,
  fix if it's infrastructure, then just re-run the same command (it resumes).
- When all three conditions finish: commit `results/tail_push/**/manifest.tsv`,
  `results/tail_push/**/eval/*.json` and `STATUS.md` (NOT logs or checkpoints)
  to `tail-push-experiment` and `git push origin tail-push-experiment`.
  Never push to main; never force-push.

## Go2 rerun (2026-11-13)
Same experiment on the Unitree Go2 (`Go2BridgeCrossing`, `Go2BridgeCrossingTail`,
code in `mujoco_playground/_src/locomotion/go2/`). Rewards and curriculum are the
same as the Go1 run except: foot height measured from the deck, per-episode info
reset, Kp 50 (the Go2 sags at Kp 35), and a pitch penalty with a +/-10 deg dead band
(scale -75). Results go to `results/tail_push_go2/<cond>/`; WandB project
`bridge_tail_push_go2`. Phase C now also runs the 0.10 m cross-width evals and the
posture diagnostics, and evals use the fine sweep (0.5 N*s, 512 trials).
```sh
cd ~/bridge_tail_push && git pull --ff-only origin tail-push-experiment
mkdir -p results/tail_push_go2
ROBOT=go2 CONDITION=active CUDA_VISIBLE_DEVICES=0 setsid nohup ./run_tail_push_experiment.sh > results/tail_push_go2/active.launcher.log 2>&1 &
ROBOT=go2 CONDITION=locked CUDA_VISIBLE_DEVICES=1 setsid nohup ./run_tail_push_experiment.sh > results/tail_push_go2/locked.launcher.log 2>&1 &
ROBOT=go2 CONDITION=notail CUDA_VISIBLE_DEVICES=0 setsid nohup ./run_tail_push_experiment.sh > results/tail_push_go2/notail.launcher.log 2>&1 &
```
When done, commit `results/tail_push_go2/**/{manifest.tsv,eval/*.json,posture/*.json}`
and `STATUS.md` (not logs/checkpoints) and push to `tail-push-experiment`.
