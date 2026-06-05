# notes/

Operational notes and experiment records for the Go1 bridge-crossing RL project (JAX PPO + MuJoCo/Brax, curriculum learning).

---

## Structure

### Top level

| File | Contents |
|------|----------|
| `progress.md` | Current project status, full curriculum results table (both no-depth and depth-obs runs), key findings (depth obs breakthrough, 0.1m physics wall), and next steps. |

### research/

| File | Contents |
|------|----------|
| `what_to_build.md` | Ordered rationale for building the environment: arena → rulebook → registration → PPO config → test → train. Explains why each step is a prerequisite for the next. |
| `problems.md` | Running log of issues encountered and their fixes: success bonus sizing, reward clip at 0 silencing penalties, NaN propagation, lateral-deviation tuning, the no-perception ceiling at 0.3m, and NCCL/IOMMU multi-GPU issues. |
| `potential_improvements.md` | Lessons captured for the next foundation run: adding y-position and foot-contact to the state obs, reward clip lower bound, success scale rule of thumb, finer curriculum staging, observation history, and exponential lateral-deviation penalty shape. |

### training/

| File | Contents |
|------|----------|
| `training_code_structure.md` | Which of the three files does what: `train_jax_ppo.py` (runner/GPU setup), `locomotion_params.py` (PPO hyperparameters), `go1/bridge.py` (environment). Includes the call-flow diagram. |
| `reading_training_results_HOWTO.md` | Where logs and checkpoints live, how to read per-term reward metrics, how to compute success rate and fall rate from raw numbers, how to resume from a checkpoint, and which WandB panels to watch. |
| `2gpu_training_HOWTO.md` | Hardware requirements, how Brax `pmap` uses two GPUs, recommended `num_envs`/`batch_size` values, required XLA/NCCL env vars, two-layer compilation cache setup (autotune pbtxt + JAX persistent cache), diagnosing stuck runs, and the IOMMU/P2P bottleneck with fix options. |
| `viewer_launch_HOWTO.md` | One-liner to open the MuJoCo passive viewer on macOS using `mjpython`. |
| `xla_jit_compilation_and_gpu_setup.md` | XLA JIT lessons: why changing `num_envs` without adjusting batch math causes multi-hour hangs, what the "configs filtered out" autotuning warning means, `MUJOCO_GL=egl` for video rendering, and the recommended smoke-test and full-training commands. |

### hardware/

| File | Contents |
|------|----------|
| `bios_fix.md` | Step-by-step fix for NCCL P2P being blocked by IOMMU+ACS on dual RTX PRO 6000 Blackwell: disable ACS in BIOS + `iommu=pt pci=realloc` kernel cmdline. Includes timing benchmarks and software fallback (`NCCL_P2P_DISABLE=1`). |
| `test_best_gpu_config.md` | TODO experiments for finding the optimal `num_envs` for 2-GPU training: doubling to 8192, stepping up to saturate VRAM, and a single-GPU benchmark to check whether pmap AllReduce overhead outweighs the second GPU. |

---

## Quick reference

| Looking for | Go to |
|-------------|-------|
| Current success rates across all curriculum stages | `progress.md` |
| Why depth obs matters / what changed between Curriculum 1 and 2 | `progress.md` — Key Findings |
| What to do differently on the next foundation run | `research/potential_improvements.md` |
| What went wrong and how it was fixed | `research/problems.md` |
| How to compute success rate from a log line | `training/reading_training_results_HOWTO.md` |
| How to resume training from a checkpoint | `training/reading_training_results_HOWTO.md` — Resuming from a checkpoint |
| Which file to edit for PPO hyperparameters vs reward design | `training/training_code_structure.md` |
| 2-GPU training is slow or hanging | `hardware/bios_fix.md`, `training/2gpu_training_HOWTO.md` — IOMMU section |
| XLA compile is taking hours / hangs with no output | `training/xla_jit_compilation_and_gpu_setup.md`, `training/2gpu_training_HOWTO.md` — What NOT to do |
| Opening the 3D viewer on macOS | `training/viewer_launch_HOWTO.md` |
| BIOS steps to fix NCCL P2P on dual Blackwell cards | `hardware/bios_fix.md` |
