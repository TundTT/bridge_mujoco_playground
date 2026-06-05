# Plan

This folder tracks implementation plans for the Go1 Bridge Crossing project — a MuJoCo/Brax RL task where a Unitree Go1 quadruped learns to traverse a narrow bridge using curriculum learning and JAX PPO.

## Stage System

Plans move through four stages:

| Stage | Meaning |
|---|---|
| `1_not_started` | Drafted but not yet started |
| `2_review` | Under review before implementation begins |
| `3_to_be_implemented` | Approved and queued for implementation |
| `4_completed` | Implementation done and verified |

## Completed Plans

Implementation order is defined in `4_completed/.order.md`.

| File | Description | Status |
|---|---|---|
| `bridge_terrain_xml.md` | MuJoCo XML scene (`scene_mjx_feetonly_bridge.xml`) with two platforms and a configurable-width bridge beam; keyframe, collision setup, camera framing. Verified in viewer 2026-05-17. | COMPLETED |
| `go1_environment_module.md` | `go1/bridge.py` — the `BridgeCrossing` environment class: bridge-width patching, reset randomisation, termination conditions, reward design (forward velocity, orientation, alive, energy, gait), and observation vector (46D state + privileged state). | COMPLETED |
| `register_environment.md` | Wiring `Go1BridgeCrossing` into `locomotion/__init__.py` and adding the PPO config block to `locomotion_params.py`; includes remaining follow-up items (`goal_reached`, `lateral_deviation`, success termination). | COMPLETED |
| `ppo_training_config.md` | PPO hyperparameter rationale (timesteps, discounting, network sizes, obs keys); curriculum commands for each bridge-width stage; training health metrics to monitor; hardware notes for dual RTX 6000. | COMPLETED |
| `train_and_iterate.md` | Placeholder for training iteration notes. | COMPLETED |
| `heightmap_generation_tests.md` | Three-stage test protocol for validating the world heightmap and local heightmap before training: Stage 1 numpy-only generation check, Stage 2 ground-truth comparison against the loaded MuJoCo model, Stage 3 visual 13x13 local patch verification. All three stages passed 2026-06-04. Key fix: `np.arange` instead of `np.linspace` for exact 3cm cell spacing. | COMPLETED |
| `local_heightmap_observation.md` | Design and implementation of the 13x13 local heightmap observation (169 floats) appended to the policy state vector (46 -> 215D): world heightmap construction, per-step robot-local sampling via yaw rotation, integration into `_get_obs()`. Resolves the perception wall at narrow bridge widths. Experiment results: 91% success at 0.3m (+42 pp), 89% at 0.2m (+73 pp). | COMPLETED |
