# How to Read Training Results

## Where everything lives

```
bridge_mujoco_playground/
├── logs/
│   └── Go1BridgeCrossing-YYYYMMDD-HHMMSS/   ← one folder per run
│       ├── checkpoints/
│       │   ├── 000039321600/                 ← checkpoint at that step
│       │   ├── 000078643200/
│       │   └── 000235929600/                 ← latest = best to resume from
│       ├── metrics_history.json              ← full eval history as JSON
│       └── rollout0.mp4                      ← video of final policy
└── wandb/
    └── run-YYYYMMDD_HHMMSS-<id>/             ← WandB local cache
```

Live log output is redirected to `/tmp/` during training, named by what the run was:
- `/tmp/bridge_0.3m_ext200.log` — the 0.3m 200M-step extension
- `/tmp/bridge_curriculum_0.3m.log` — the original 0.3m curriculum run
- `/tmp/bridge_log_reward.log` — etc.

---

## How to tell Claude a run finished

Just say "run finished" or paste the final lines. Claude will read the log automatically.

To read it yourself, the key lines are near the bottom:

```
235929600: reward=58.486          ← step number : total episode reward
  reward/success: 2421.87500      ← raw metric value (see below)
  reward/termination: -0.52344
  reward/forward_vel: 346.87946
  ...
Done training.
```

---

## How to interpret the numbers

### Total reward
The number after `reward=` is the mean episode reward across all eval envs. Higher = better, but the absolute value depends on reward scales so comparisons only make sense within the same run or between runs with identical configs.

### Per-term metrics

Each `reward/<term>` is: `raw_value × scale × dt` summed over the episode.

The scales are in `bridge.py` → `default_config()` → `reward_config.scales`. Current values:

| Term | Scale | What it means |
|------|-------|---------------|
| `success` | +5000 | Times scale (5000) × fraction of envs that succeeded |
| `termination` | -1.0 | Negative = fell; abs value ≈ fall rate |
| `forward_vel` | +2.0 | How fast robot moved in +x direction |
| `progress_to_goal` | +3.0 | Log-shaped reward for x-position progress |
| `lateral_deviation` | -3.0 | Penalty for y-offset from bridge centre |
| `heading` | -2.0 | Penalty for not facing +x |
| `orientation` | -5.0 | Penalty for tilting (roll/pitch) |
| `alive` | +0.1 | Reward per timestep just for surviving |
| `energy` | -0.001 | Penalty for joint torque × velocity |
| `torques` | -0.0002 | Penalty for large joint torques |
| `action_rate` | -0.01 | Penalty for jerky actions |
| `feet_air_time` | +0.1 | Encourages proper stepping gait |

### Success rate (most important)

```
success_rate = reward/success  /  5000
```

Example: `reward/success: 2421.88` → `2421.88 / 5000 = 48.4%` of eval episodes succeeded.

### Fall rate

```
fall_rate = abs(reward/termination)
```

Example: `reward/termination: -0.52344` → 52.3% fall rate.

Success + fall rate should sum to ~100% (the remainder are timeouts — robot survived but didn't reach the goal).

---

## Resuming from a checkpoint

To continue training from where a run left off:

```bash
train-jax-ppo \
  --env_name Go1BridgeCrossing \
  --num_timesteps 200000000 \
  --load_checkpoint_path logs/<run-folder>/checkpoints/<step-number> \
  --playground_config_overrides '{"bridge_half_width": 0.15}' \
  --wandb_project bridge_crossing_1 \
  --wandb_run_name my_run_name \
  --use_wandb \
  > /tmp/my_run.log 2>&1 &
```

Find the latest checkpoint step:
```bash
ls logs/<run-folder>/checkpoints/ | sort -n | tail -1
```

---

## Curriculum chain so far

| Stage | bridge_half_width | Width | Steps | Final success |
|-------|-------------------|-------|-------|---------------|
| Foundation | 0.4 | 0.8m | 300M | ~77% |
| 0.5m | 0.25 | 0.5m | 250M | ~65% |
| 0.4m | 0.2 | 0.4m | 250M | ~68% |
| 0.3m | 0.15 | 0.3m | ~436M total | ~48–52% (plateaued) |

Note: `bridge_half_width` in config is the half-extent — multiply by 2 for actual bridge width.

---

## WandB

All runs stream to: https://wandb.ai/Tund/bridge_crossing_1

Useful panels to watch:
- `eval/episode_reward` — total reward curve
- `reward/success` — divide by 5000 for success rate
- `reward/termination` — abs value = fall rate
- `reward/forward_vel` — is the robot actually moving forward?
