# Phase: PPO Training Config

> **Status: IMPLEMENTATION COMPLETED**
>
> `bridge.py` is complete and `Go1BridgeCrossing` is registered. The only blocker before
> `train-jax-ppo --env_name Go1BridgeCrossing` works is adding the env to
> `locomotion_params.py`. Without it the script raises `ValueError: Unsupported env: Go1BridgeCrossing`.

---

## What Needs to Change

One file: `mujoco_playground/config/locomotion_params.py`

Add an `elif` branch to `brax_ppo_config()` for `"Go1BridgeCrossing"` before the final `else: raise ValueError` line (currently line 162).

```python
elif env_name == "Go1BridgeCrossing":
    rl_config.num_timesteps = 100_000_000
    rl_config.num_evals = 10
    rl_config.discounting = 0.99
    rl_config.entropy_cost = 0.01
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
```

---

## Rationale for Each Hyperparameter

| Parameter | Value | Reason |
|---|---|---|
| `num_timesteps` | 100M | Bridge crossing is less complex than joystick command tracking, and roughly comparable to `Go1Handstand`. Use Go1Handstand's budget as the baseline; extend to 200M if the policy plateaus. |
| `num_evals` | 10 | One checkpoint every 10M steps — enough resolution to catch reward plateaus early |
| `discounting` | 0.99 | Longer horizon than the 0.97 default. The bridge is ~4m and takes ~200 steps at the robot's natural pace; the robot must credit reaching Platform B back from up to 200 steps away. 0.97^200 ≈ 0.002 — too much discounting wipes out the goal signal. |
| `entropy_cost` | 0.01 | Default value. Bridge task requires exploration to discover that walking forward is safe — leave entropy encouragement on until the policy is clearly converging. |
| `policy_hidden_layer_sizes` | (512, 256, 128) | Same as all other Go1 tasks. Policy obs is 46D — no need for wider layers. |
| `value_hidden_layer_sizes` | (512, 256, 128) | Asymmetric AC: critic sees `privileged_state` (~70D). Same size as policy is fine for this scale. |
| `policy_obs_key` | `"state"` | Policy only gets the noisy 46D obs — matches real-hardware deploy. |
| `value_obs_key` | `"privileged_state"` | Critic gets full noiseless info during training only. This is the standard asymmetric AC pattern used by all other Go1 envs. |
| `reward_scaling` | 1.0 (default) | The reward is already clipped to `[0, 10000]` in `bridge.py:step()`. No additional scaling needed. |
| `num_envs` | 8192 (default) | Standard for locomotion envs. Reduce to 4096 if VRAM is tight. |
| `unroll_length` | 20 (default) | 20 steps × 0.02s = 0.4s of rollout per update. Covers roughly one gait cycle. |
| `num_minibatches` | 32 (default) | Standard for this batch size. |
| `max_grad_norm` | 1.0 (default) | Keeps training stable with the narrow platform termination spikes. |

---

## Curriculum: How to Use `bridge_half_width`

The `bridge_half_width` env config field (default 0.2 = 0.4m full width) is the primary curriculum dial. To narrow the bridge, pass it as a `playground_config_overrides` override at launch:

```sh
# Stage 1 — wide bridge, easy (0.8m wide)
train-jax-ppo --env_name Go1BridgeCrossing \
  --playground_config_overrides '{"bridge_half_width": 0.4}'

# Stage 2 — medium (0.4m wide, default)
train-jax-ppo --env_name Go1BridgeCrossing

# Stage 3 — narrow (0.2m wide)
train-jax-ppo --env_name Go1BridgeCrossing \
  --playground_config_overrides '{"bridge_half_width": 0.1}'

# Fine-tune from a checkpoint
train-jax-ppo --env_name Go1BridgeCrossing \
  --load_checkpoint_path /path/to/stage2/checkpoint \
  --playground_config_overrides '{"bridge_half_width": 0.1}'
```

**Curriculum advancement criterion:** advance to the next stage when mean episodic return
stabilises (< 5% change over 3 consecutive evals) AND the agent successfully reaches
Platform B in > 60% of eval episodes.

---

## Hardware

**Available:** 2× NVIDIA RTX 6000, 98 GB VRAM each.

**Strategy:** run standard config first (smoke test → full Stage 1 run). Only tune for hardware after the standard run validates the config end-to-end.

**Hardware optimisation (post-validation):**
- Increase `num_envs` to `32768` or `65536` — VRAM is not the constraint.
- Check multi-GPU support: `python -c "import jax; print(jax.devices())"`. If both GPUs show up and the trainer uses `pmap`, envs are split automatically — set `num_envs` to a multiple of 2 (e.g. `16384`).
- Parallel curriculum: run Stage 1 and Stage 2 simultaneously, one job per GPU.

---

## Training Commands

```sh
cd bridge_mujoco_playground
source .venv/bin/activate

# Smoke test: 2M steps, 512 envs (verifies the config path doesn't error)
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 2000000 --num_envs 512 --num_evals 5

# Full run (Stage 1, wide bridge)
train-jax-ppo --env_name Go1BridgeCrossing \
  --playground_config_overrides '{"bridge_half_width": 0.4}'

# With WandB logging
train-jax-ppo --env_name Go1BridgeCrossing --use_wandb \
  --playground_config_overrides '{"bridge_half_width": 0.4}'

# Hardware-optimised run (post smoke test validation only)
train-jax-ppo --env_name Go1BridgeCrossing --num_envs 32768 \
  --playground_config_overrides '{"bridge_half_width": 0.4}'
```

---

## What to Watch During Training

**Reward components to monitor** (logged as `eval/reward/<key>` metrics):

| Metric | Healthy sign | Warning sign |
|---|---|---|
| `reward/forward_vel` | Grows steadily in early training | Flat or negative — robot is not learning to walk forward |
| `reward/alive` | Should dominate early reward | Collapses abruptly — robot is falling frequently |
| `reward/termination` | Decreases in magnitude over time | Stays large — persistent falling |
| `reward/orientation` | Small and decreasing | Large and growing — robot is tumbling |
| `reward/feet_air_time` | Non-zero and positive | Zero — robot is sliding (not gait-stepping) |
| `reward/energy` | Small decrease | Explodes — likely a PD gain or action scale issue |

**Episode length** (`eval/episode_length`): should increase from ~100 steps (early falls) toward the full 1000-step budget as the policy improves.

**Fall rate**: if `reward/termination` stays near `-1.0` (max) past 20M steps, the robot is still falling immediately. Check `fall_threshold` in the env config — it may be triggering spuriously.

---

## Potential Issues and Fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ValueError: Unsupported env: Go1BridgeCrossing` | PPO config not added | This phase — add the elif branch |
| `KeyError: 'Go1BridgeCrossing'` in registry | `locomotion/__init__.py` missing entry | Already done (Phase: Register Environment) |
| Episode never reaches Platform B | Reward signal too sparse | Add `goal_reached` sparse bonus to `bridge.py:_get_reward()` (noted as [TO ADD] in env plan) |
| Robot hugs one edge and falls on narrow bridge | No centreline incentive | Add `lateral_deviation` penalty to `bridge.py:_get_reward()` (noted as [TO ADD] in env plan) |
| Training explodes (NaN loss) | `discounting` too high or `learning_rate` too high | Drop `discounting` to 0.97, halve `learning_rate` |
| Very slow convergence | `entropy_cost` too high | Drop `entropy_cost` to 0.005 after 30M steps |

---

## Missing Reward Terms (From Environment Plan)

Two reward terms are marked `[TO ADD]` in `go1_environment_module.md`. They are not
blockers for the PPO config but should be added to `bridge.py` before or alongside this
phase if initial training shows the symptoms above:

1. **`goal_reached`** — one-time sparse `+5.0` bonus when `data.qpos[0] > platform_b_x`.
   Add to `_get_reward()` and to `reward_config.scales` in `default_config()`.
   The `platform_b_x` threshold is approximately `x = 5.0m` based on the bridge XML geometry.

2. **`lateral_deviation`** — `−data.qpos[1]² * scale`. Add as a small negative term
   (scale ≈ −0.5) to discourage the robot from drifting off the bridge centreline.

Both can be added in a single follow-up edit to `bridge.py` without touching the PPO config.
