# Project Progress

## Status: Depth Curriculum Complete — 0.2m Solved, 0.1m Physics Wall Confirmed

---

## What Has Been Built

- `mujoco_playground/_src/locomotion/go1/bridge.py` — `Go1BridgeCrossing` environment
- Registered as `Go1BridgeCrossing` in the locomotion registry
- PPO config added in `mujoco_playground/config/locomotion_params.py`
- Training script extended with per-term reward logging, WandB project/run name flags, metrics history JSON export, **per-eval rollout video logging to WandB**
- **Local heightmap observation** (169 floats, 13×13 patch @ 3cm/cell, robot-local frame) added to `state` observation — obs size 46 → 215

Key environment design decisions:
- Log-shaped progress reward: steep gradient near spawn, tapers toward goal
- Success termination with large bonus (scale=5000) to prevent goal avoidance
- Reward clipped to `[-10, 10000]` (not 0) so penalty signals remain active
- Heading penalty on y-component of forward vector (not raw yaw)
- See `notes/research/problems.md` for issues hit and fixes applied

---

## Curriculum 1 — No Depth Obs (baseline)

WandB project: `bridge_crossing_1`

| Stage | Width | Steps | WandB run | Success | Fail | Checkpoint |
|-------|-------|-------|-----------|---------|------|------------|
| Foundation | 0.8m | 300M | `curriculum_0.6m` | ~77% | ~23% | `logs/Go1BridgeCrossing-20260524-045220/checkpoints/000353894400` |
| 0.5m | 0.5m | 236M | `curriculum_0.5m` | ~65% | ~35% | `logs/Go1BridgeCrossing-20260524-071635/checkpoints/000235929600` |
| 0.4m | 0.4m | 236M | `curriculum_0.4m` | ~68% | ~32% | `logs/Go1BridgeCrossing-20260524-073035/checkpoints/000235929600` |
| 0.3m | 0.3m | 436M | `curriculum_0.3m` + ext | ~49% | ~51% | `logs/Go1BridgeCrossing-20260524-162105/checkpoints/000235929600` |
| 0.2m | 0.2m | 236M | `curriculum_0.2m` | ~16% | ~84% | `logs/Go1BridgeCrossing-20260524-164213/checkpoints/000235929600` |
| 0.1m | 0.1m | 500M | `curriculum_0.1m` + ext | 0% | ~100% | `logs/Go1BridgeCrossing-20260526-140437/checkpoints/000353894400` |

**Ceiling:** 0.3m (~50/50). Below 0.2m the robot had no edge signal and couldn't cross.

---

## Curriculum 2 — With Depth Obs (local heightmap, 2026-06-04)

WandB project: `bridge_crossing_with_depth`

Observation: 13×13 terrain patch in robot-local frame appended to state (169 floats).
The patch rotates with the robot's yaw so the policy always sees ahead/left/right edges directly.

| Stage | Width | Steps | WandB run | Success | Fail | Checkpoint |
|-------|-------|-------|-----------|---------|------|------------|
| 0.8m | 0.8m | 300M | `depth_0.8m` | 96% | 4% | `logs/Go1BridgeCrossing-20260604-143931/checkpoints/000353894400` |
| 0.7m | 0.7m | 200M | `depth_0.7m` | 100% | 0% | `logs/Go1BridgeCrossing-20260604-145237/checkpoints/000235929600` |
| 0.6m | 0.6m | 200M | `depth_0.6m` | 100% | 0% | `logs/Go1BridgeCrossing-20260604-150313/checkpoints/000235929600` |
| 0.5m | 0.5m | 200M | `depth_0.5m` | 100% | 0% | `logs/Go1BridgeCrossing-20260604-*/checkpoints/000235929600` |
| 0.4m | 0.4m | 200M | `depth_0.4m` | 98% | 2% | `logs/Go1BridgeCrossing-20260604-*/checkpoints/000235929600` |
| 0.3m | 0.3m | 200M | `depth_0.3m` | **91%** | 9% | `logs/Go1BridgeCrossing-20260604-*/checkpoints/000235929600` |
| 0.2m | 0.2m | 200M | `depth_0.2m` | **89%** | 11% | `logs/Go1BridgeCrossing-20260604-*/checkpoints/000235929600` |
| 0.1m | 0.1m | 300M | `depth_0.1m` | 0% | 100% | `logs/Go1BridgeCrossing-20260604-*/checkpoints/000353894400` |

Note: `bridge_half_width` in config = half the physical width (e.g. 0.15 → 0.3m bridge).

---

## Key Findings

### Depth obs breakthrough
- The perception wall is broken: 0.3m jumped from **49% → 91%** and 0.2m from **16% → 89%**
- 0.6m and 0.7m reached **100% success, 0% falls** — the robot now reliably centres itself
- The local heightmap patch gives the policy a direct edge-drift signal it previously lacked
- Each stage converges faster and to a higher level than the equivalent no-depth run

### 0.1m is a physics wall, not a perception wall
- Robot stance width ≈ 0.28m; the 0.1m bridge is narrower than the distance between feet
- At 0.1m the robot cannot place all four feet on the bridge simultaneously
- 300M steps confirmed: 0% success, 100% falls — more training will not help
- A fundamentally different strategy is needed to cross 0.1m (e.g. sideways gait, bipedal stance, or a different robot)

### Heightmap implementation details
- Grid: `np.arange` (not `linspace`) for exact 0.03m spacing → shape (334, 68)
- Local patch rotates with robot yaw via `forward = site_xmat @ [1,0,0]`, normalised
- Built once at `__init__` after bridge-width patch; rebuilt each curriculum stage
- Stage 2 ground-truth check: 0/22,712 cells disagree with MuJoCo model
- See `.workspace/Plan/1_not_started/test_out/` for all test plots and results

---

## Next Steps

- **0.2m is now near-solved (89%)** — could push further with more steps or reward tuning
- **0.1m requires a different approach:** sideways crab-walk gait, reduced stance width, or a narrower robot (humanoid with smaller foot separation)
- Videos of every eval checkpoint are logged to WandB `bridge_crossing_with_depth` — review to see qualitative gait changes across widths
- Consider testing on humanoid (H1/G1) which may have a narrower effective stance
