# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Before beginning any work in this project, read @Task.md to understand the task context and objectives.

For current project progress and curriculum training results, see `.workspace/notes/research/progress.md`.

---

# Development Environment

## Platform split

- **Development machine**: macOS (where this code is written and Claude Code runs)
- **Target machine**: Linux workstation (where the code actually executes)

## Testing policy

At the start of every new conversation, run the following to detect the environment before doing any work:

```sh
uname -s && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
```

Interpret the result:
- **Darwin**: Mac — do not run anything requiring the workstation (MuJoCo simulation, GPU training, Linux-only tools). Acknowledge when something cannot be verified locally.
- **Linux + two RTX 6000 GPUs listed**: Confirmed workstation — full access, all training and simulation commands available.
- **Linux + anything else**: Unknown Linux machine — treat as Mac (no GPU training, no simulation) unless the user explicitly confirms otherwise.

---

## Project Overview

This repo contains a fork of Google DeepMind's `mujoco_playground` library, located in `bridge_mujoco_playground/`. The goal is to implement a **Narrow Bridge Locomotion Task** (see Task.md) — a new MJX environment where a legged robot (Right now Unitree Go1 quadruped but will also be used for humanoids robots) must traverse a narrow bridge of configurable width between two platforms, supporting curriculum learning.

All work happens inside `bridge_mujoco_playground/`. Run commands from that directory.

---

## Development Commands

All commands run from `bridge_mujoco_playground/`.

**Linting & formatting:**
```sh
ruff check .
ruff format .
```

**Type checking:**
```sh
mypy mujoco_playground/
```

**Tests:**
```sh
pytest mujoco_playground/_src/                        # all tests
pytest mujoco_playground/_src/locomotion/locomotion_test.py  # locomotion only
pytest mujoco_playground/_src/ -k "test_name"         # single test
```

**Training (JAX PPO):**
```sh
train-jax-ppo --env_name Go1JoystickFlatTerrain
train-jax-ppo --env_name Go1JoystickFlatTerrain --impl warp  # MuJoCo Warp backend
train-jax-ppo --env_name CartpoleBalance --num_timesteps 500000 --num_envs 512
```

**Bridge crossing training:**
```sh
# Fresh run
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 300000000 \
  --playground_config_overrides '{"bridge_half_width": 0.4}' \
  --wandb_project bridge_crossing_1 \
  --wandb_run_name my_run_name \
  --use_wandb \
  > /tmp/my_run.log 2>&1 &

# Resume from checkpoint (curriculum)
train-jax-ppo --env_name Go1BridgeCrossing \
  --num_timesteps 200000000 \
  --load_checkpoint_path logs/<run-folder>/checkpoints/<step> \
  --playground_config_overrides '{"bridge_half_width": 0.25}' \
  --wandb_project bridge_crossing_1 \
  --wandb_run_name curriculum_0.5m \
  --use_wandb \
  > /tmp/my_run.log 2>&1 &
```

WandB project: `bridge_crossing_1` (https://wandb.ai/Tund/bridge_crossing_1)
Logs & checkpoints: `bridge_mujoco_playground/logs/<run-folder>/`

**Training (RSL-RL PPO):**
```sh
train-rsl-ppo --env_name Go1JoystickFlatTerrain
```

---

## Setup & Installation

```sh
cd bridge_mujoco_playground
uv venv --python 3.12
source .venv/bin/activate
uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
uv --no-config sync --all-extras
```

Verify: `python -c "import mujoco_playground; print('Success')"`

Robot assets (mujoco_menagerie) are downloaded automatically on first `load()` call.

---

## Architecture

### Core abstractions (`mujoco_playground/_src/`)

- **`mjx_env.py`** — `MjxEnv` abstract base class. All environments subclass this. Key methods: `reset(rng)`, `step(state, action)`. Key properties: `xml_path`, `action_size`, `mj_model`, `mjx_model`. Also contains `State` (flax struct: `data, obs, reward, done, metrics, info`), `update_assets()`, `ensure_menagerie_exists()`.

- **`registry.py`** — Top-level `load(env_name, config, config_overrides)` and `get_default_config(env_name)`. Delegates to `locomotion`, `manipulation`, or `dm_control_suite` sub-registries.

- **`reward.py`** — JAX port of `dm_control.utils.rewards` (tolerance functions, sigmoid shaping).

- **`wrapper.py`** / **`wrapper_torch.py`** — Gym-like wrappers (Brax training wrapper, PyTorch wrapper).

- **`gait.py`** — Gait phase utilities for locomotion rewards.

### Locomotion environments (`mujoco_playground/_src/locomotion/`)

Each robot has its own subdirectory (e.g. `go1/`, `h1/`, `g1/`, `spot/`) with a consistent structure:
- `<robot>_constants.py` — paths to XML files, sensor names, joint names, body names
- `base.py` — `<Robot>Env(MjxEnv)` loading the MjModel, setting PD gains, defining sensor helpers
- `joystick.py` (or task-specific file) — full environment: `default_config()`, `reset()`, `step()`, observation & reward computation
- `randomize.py` — `domain_randomize(model, rng)` for sim-to-real
- `xmls/` — MuJoCo XML scene files (terrain, robot composition)

The locomotion sub-registry in `mujoco_playground/_src/locomotion/__init__.py` maps string env names to classes and exposes `register_environment(name, cls, cfg_fn)` to add new envs without modifying the registry core.

### Config system (`mujoco_playground/config/`)

`locomotion_params.py` provides `brax_ppo_config(env_name)` — tuned PPO hyperparameters per environment. Environment-level configs use `ml_collections.config_dict.ConfigDict` returned by each env's `default_config()`.

### Training scripts (`learning/`)

- `train_jax_ppo.py` — Brax PPO trainer. Entry point via `train-jax-ppo` CLI. Supports WandB, TensorBoard, rscope visualization, checkpointing.
- `train_rsl_rl.py` — RSL-RL PPO trainer. Entry point via `train-rsl-ppo` CLI.

---