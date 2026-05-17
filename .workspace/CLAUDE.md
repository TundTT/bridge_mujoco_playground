# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Before beginning any work in this project, read @Task.md to understand the task context and objectives.

Implementation plans are kept in @Plan/ — note these are subject to change.

## Project Overview

This repo contains a fork of Google DeepMind's `mujoco_playground` library, located in `bridge_mujoco_playground/`. The goal is to implement a **Narrow Bridge Locomotion Task** (see Task.md) — a new MJX environment where a legged robot (Unitree Go2 quadruped or Unitree H1 humanoid) must traverse a narrow bridge of configurable width between two platforms, supporting curriculum learning.

All work happens inside `bridge_mujoco_playground/`. Run commands from that directory.

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

**Training (RSL-RL PPO):**
```sh
train-rsl-ppo --env_name Go1JoystickFlatTerrain
```

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

## Adding the Bridge Task

The bridge environment should follow the pattern of existing locomotion tasks. The recommended approach:

1. **Create `mujoco_playground/_src/locomotion/go2/`** (for Go2) or reuse `go1/` adapted for Go2 from menagerie, with:
   - `go2_constants.py` — XML paths, sensor/joint names
   - `base.py` — `Go2Env(MjxEnv)`
   - `bridge.py` — `BridgeEnv` with `default_config()` (include `bridge_width` param), `reset()`, `step()`, reward for forward progress + fall termination

2. **Create the bridge XML** in `xmls/scene_mjx_bridge.xml` — compose the robot MJCF with a parametric terrain (two platforms + narrow beam). Bridge width can be a parameter set at model-load time or via geom size modification in `reset()`.

3. **Register** in `mujoco_playground/_src/locomotion/__init__.py` using `register_environment()` or by adding to `_envs`/`_cfgs` dicts.

4. **Add PPO config** in `mujoco_playground/config/locomotion_params.py`.

Robot XML assets are in `mujoco_playground/external_deps/mujoco_menagerie/` (auto-downloaded). Use `mjx_env.MENAGERIE_PATH / "unitree_go2"` for Go2 and `mjx_env.MENAGERIE_PATH / "unitree_h1"` for H1.
