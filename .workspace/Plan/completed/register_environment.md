# Phase: Register Environment

> **Status: IMPLEMENTATION COMPLETED**
>
> `Go1BridgeCrossing` is registered in `locomotion/__init__.py` and the PPO config block
> is present in `locomotion_params.py`. The env loads and the training command resolves
> without error. Three reward/termination items remain as follow-up work in `bridge.py`.

---

## What Was Done

### 1. Import and registry entries — `locomotion/__init__.py`

```python
# Import (line 31)
from mujoco_playground._src.locomotion.go1 import bridge as go1_bridge

# _envs dict (line 69)
"Go1BridgeCrossing": go1_bridge.BridgeCrossing,

# _cfgs dict (line 104)
"Go1BridgeCrossing": go1_bridge.default_config,
```

No `_randomizer` entry — `Go1BridgeCrossing` intentionally omits domain randomisation
for the initial training run. If sim-to-real becomes a goal, add
`"Go1BridgeCrossing": go1_randomize.domain_randomize` following the joystick pattern.

### 2. PPO config block — `locomotion_params.py` (lines 151–161)

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

See `ppo_training_config.md` for the rationale behind each hyperparameter value.

---

## Remaining Work in `bridge.py`

Three items are marked `[TO ADD]` in `go1_environment_module.md`. They are not blockers
for the registration or smoke-test run but must be added before the policy can learn the
full task.

### A. `goal_reached` reward term

Add to `_get_reward()` in [bridge.py](mujoco_playground/_src/locomotion/go1/bridge.py):

```python
"goal_reached": self._reward_goal_reached(data),
```

Add the helper:

```python
def _reward_goal_reached(self, data: mjx.Data) -> jax.Array:
    return jp.float32(data.qpos[0] > 5.0)
```

Add to `reward_config.scales` in `default_config()`:

```python
goal_reached=5.0,
```

The threshold `5.0` is the start of Platform B (`x = 4.0` to `x = 7.0`; `x = 5.0` is
well clear of the bridge–platform seam). Adjust if the sparse signal fires too early.

### B. `lateral_deviation` penalty

Add to `_get_reward()`:

```python
"lateral_deviation": self._cost_lateral_deviation(data),
```

Add the helper:

```python
def _cost_lateral_deviation(self, data: mjx.Data) -> jax.Array:
    return jp.square(data.qpos[1])
```

Add to `reward_config.scales` in `default_config()`:

```python
lateral_deviation=-0.5,
```

This penalises the robot for drifting off the bridge centreline (`y = 0`). Without it
the robot has no incentive to stay centred and may learn to hug one edge — a strategy
that fails when the bridge narrows in later curriculum stages.

### C. Success termination

Extend `_get_termination()` to end the episode when the robot reaches Platform B:

```python
def _get_termination(self, data: mjx.Data) -> jax.Array:
    tilt_termination = self.get_upvector(data)[-1] < 0.0
    height_termination = data.qpos[2] < self._config.fall_threshold
    goal_reached = data.qpos[0] > 5.0
    return tilt_termination | height_termination | goal_reached
```

Without this the robot keeps walking for the remaining steps after reaching Platform B,
diluting the rollout with uninformative transitions. The threshold should match the one
used for the `goal_reached` reward term so they fire at the same moment.

---

## Verification Checklist

- [x] `mujoco_playground.locomotion.load("Go1BridgeCrossing")` does not raise
- [x] `mujoco_playground.config.locomotion_params.brax_ppo_config("Go1BridgeCrossing")` does not raise
- [x] Smoke test command resolves without `ValueError: Unsupported env`
- [ ] `goal_reached`, `lateral_deviation`, `success termination` added to `bridge.py`
- [ ] Smoke test run completes (workstation only — cannot verify on Mac)

```sh
# Smoke test (run on workstation)
cd bridge_mujoco_playground && source .venv/bin/activate
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 2000000 --num_envs 512 --num_evals 5
```
