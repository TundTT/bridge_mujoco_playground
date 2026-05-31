# Project Progress

## Status: Curriculum Phase 1 Complete — Perception Improvements Needed

---

## What Has Been Built

- `mujoco_playground/_src/locomotion/go1/bridge.py` — `Go1BridgeCrossing` environment
- Registered as `Go1BridgeCrossing` in the locomotion registry
- PPO config added in `mujoco_playground/config/locomotion_params.py`
- Training script extended with per-term reward logging, WandB project/run name flags, metrics history JSON export

Key environment design decisions:
- Log-shaped progress reward: steep gradient near spawn, tapers toward goal
- Success termination with large bonus (scale=5000) to prevent goal avoidance
- Reward clipped to `[-10, 10000]` (not 0) so penalty signals remain active
- Heading penalty on y-component of forward vector (not raw yaw)
- See `notes/research/problems.md` for issues hit and fixes applied

---

## Curriculum Training Results

All runs in WandB project: `bridge_crossing_1`

| Stage | Width | Steps | WandB run | Final success | Fall rate | Checkpoint |
|-------|-------|-------|-----------|--------------|-----------|------------|
| Foundation | 0.8m | 300M | `foundation` | ~77% | ~23% | `logs/Go1BridgeCrossing-20260524-045220/checkpoints/000353894400` |
| 0.5m | 0.5m | 250M | `curriculum_0.5m` | ~65% | ~35% | `logs/Go1BridgeCrossing-20260524-071635/checkpoints/...` |
| 0.4m | 0.4m | 250M | `curriculum_0.4m` | ~68% | ~32% | `logs/Go1BridgeCrossing-20260524-073035/checkpoints/...` |
| 0.3m | 0.3m | 436M total | `curriculum_0.3m` + `curriculum_0.3m_ext200` | ~49% | ~51% | `logs/Go1BridgeCrossing-20260524-162105/checkpoints/000235929600` |
| 0.2m | 0.2m | 236M | `curriculum_0.2m` | ~16% | ~84% | `logs/Go1BridgeCrossing-20260524-164213/checkpoints/000235929600` |
| 0.1m | 0.1m | 500M total | `curriculum_0.1m` + `curriculum_0.1m_ext300` | 0% | ~5% | `logs/Go1BridgeCrossing-20260526-133952/checkpoints/000235929600` |

Note: `bridge_half_width` in config = half the physical width (e.g. 0.15 → 0.3m bridge).

---

## Key Findings

- Curriculum learning works: each stage warm-starts with meaningfully higher success than training from scratch at that width
- 0.3m is the practical ceiling with the current observation set (~50/50 coin flip)
- Below 0.2m the robot stance (~0.28m) is wider than the bridge — physically impossible without edge sensing
- At 0.1m the robot learned to tightrope-walk (2–5% fall rate, good forward velocity) but never crossed — 500M steps confirmed this is an information problem, not a time problem
- Grokking was tested and did not occur

---

## Blockers / Next Steps

The curriculum has hit the perception wall. See `notes/research/potential_improvements.md` for the full list. 

## Next step.

Discussion required to figure out perception/information problem

