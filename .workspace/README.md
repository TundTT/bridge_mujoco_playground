# .workspace

Planning documents, research notes, and operational guides for the **Narrow Bridge Locomotion** project — a MuJoCo/Brax environment where a Unitree Go1 quadruped learns to cross a narrow bridge of configurable width using curriculum learning.

The implementation lives in `../bridge_mujoco_playground/`. This folder contains everything that supports it: task definition, implementation plan, training notes, and experiment results.

---

## Structure

```
.workspace/
├── Task.md                        # Task definition: terrain layout, parameters, objectives
├── Plan/                          # Implementation plan (kanban-style)
│   ├── 1_not_started/             # Queued work items
│   ├── 2_review/                  # Under review
│   ├── 3_to_be_implemented/       # Ready to implement
│   └── 4_completed/               # Done — kept for reference
│       ├── bridge_terrain_xml.md
│       ├── go1_environment_module.md
│       ├── register_environment.md
│       ├── ppo_training_config.md
│       ├── train_and_iterate.md
│       ├── heightmap_generation_tests.md   # test scripts for the local heightmap obs
│       └── local_heightmap_observation.md
└── notes/
    ├── progress.md                # Current status and curriculum results (start here)
    ├── research/
    │   ├── what_to_build.md       # Why each implementation step exists
    │   ├── problems.md            # Issues hit and fixes applied
    │   └── potential_improvements.md
    ├── training/
    │   ├── 2gpu_training_HOWTO.md           # Dual-GPU setup, NCCL config, compile caching
    │   ├── reading_training_results_HOWTO.md
    │   ├── training_code_structure.md
    │   └── viewer_launch_HOWTO.md
    └── hardware/
        ├── bios_fix.md                      # ACS/IOMMU fix for PCIe P2P performance
        ├── test_best_gpu_config.md
        └── xla_jit_compilation_and_gpu_setup.md
```

---

## Where to start

| Question | File |
|---|---|
| What is the task? | `Task.md` |
| What has been built and how did training go? | `notes/progress.md` |
| What were the implementation steps and why? | `notes/research/what_to_build.md` |
| How do I run training on two GPUs? | `notes/training/2gpu_training_HOWTO.md` |
| How do I verify the heightmap is correct? | `Plan/4_completed/heightmap_generation_tests.md` |
| What bugs were hit and how were they fixed? | `notes/research/problems.md` |

---

## Current status (as of 2026-06-04)

**Curriculum 2 (with local heightmap observation) is complete.**

The local heightmap — a 13×13 terrain patch in robot-local frame appended to the state observation — broke the perception wall that stopped Curriculum 1 at 0.2m.

| Width | Curriculum 1 (no depth) | Curriculum 2 (with depth) |
|-------|------------------------|--------------------------|
| 0.8m  | 77%                    | 96%                      |
| 0.5m  | 65%                    | 100%                     |
| 0.3m  | 49%                    | **91%**                  |
| 0.2m  | 16%                    | **89%**                  |
| 0.1m  | 0%                     | 0% (physics wall — robot stance width ≈ 0.28m) |

The 0.1m bridge is narrower than the Go1's stance; a different approach is needed (sideways gait, narrower robot, or humanoid).

WandB: [bridge_crossing_with_depth](https://wandb.ai/Tund/bridge_crossing_with_depth)
