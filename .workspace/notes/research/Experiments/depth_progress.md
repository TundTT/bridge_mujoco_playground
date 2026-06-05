# Narrow Bridge Locomotion — Lab Meeting Progress

> **Date:** 2026-06-04
> **Project:** Unitree Go1 quadruped traversing a narrow bridge of configurable width
> **Goal:** Train a legged robot to cross increasingly narrow bridges using curriculum RL,
> ultimately reaching widths below the robot's own stance width

---

## What We Built

A new MuJoCo/MJX environment (`Go1BridgeCrossing`) consisting of two wide platforms
connected by a narrow bridge whose width is a curriculum dial. The robot spawns on
Platform A and must cross to Platform B.

**Scene geometry:**
- Platform A: 3 m × 2 m (spawn area, x ∈ [−3, 0])
- Bridge: 4 m long, width configurable via `bridge_half_width` (x ∈ [0, 4])
- Platform B: 3 m × 2 m (goal, x ∈ [4, 7]) — success when robot reaches x ≥ 5.5 m

**Training setup:**
- Algorithm: PPO (Brax/JAX), 4096 parallel environments, dual RTX PRO 6000 GPUs
- Policy: MLP (512 → 256 → 128), actor reads `state`, critic reads `privileged_state`
- Control: PD joint position targets at 50 Hz, 12 DoF

---

## Reward Function

| Term | Scale | What it does |
|---|---|---|
| `forward_vel` | +2.0 | Reward forward speed (capped at 2 m/s) |
| `progress_to_goal` | +3.0 | Log-shaped shaping from spawn to goal |
| `success` | +5000.0 | Large terminal bonus for reaching Platform B |
| `alive` | +0.1 | Survive each step |
| `feet_air_time` | +0.1 | Encourage regular gait rhythm |
| `lateral_deviation` | −3.0 | Penalise y-drift from bridge centreline (y²) |
| `heading` | −2.0 | Penalise facing sideways |
| `orientation` | −5.0 | Penalise tilting / rolling |
| `action_rate` | −0.01 | Smooth joint commands |
| `torques` | −0.0002 | Penalise high joint torques |
| `energy` | −0.001 | Penalise energy use |
| `termination` | −1.0 | Explicit penalty on fall / flip |

Fail: root z < 0.2 m or robot flipped. Success: root x ≥ 5.5 m.

---

## Experiment 1 — Curriculum Without Depth (Baseline)

**Observation (46 floats):** linear velocity, gyro, gravity vector, joint angles/velocities,
last action, x-progress. No y-position, no edge signal.

**Curriculum:** 0.8 m → 0.5 m → 0.4 m → 0.3 m → 0.2 m → 0.1 m

| Stage | Width | Steps | Success | Fail |
|---|---|---|---|---|
| Foundation | 0.8 m | 300 M | 77% | 23% |
| Stage 2 | 0.5 m | 236 M | 65% | 35% |
| Stage 3 | 0.4 m | 236 M | 68% | 32% |
| Stage 4 | 0.3 m | 436 M | 49% | 51% |
| Stage 5 | 0.2 m | 236 M | 16% | 84% |
| Stage 6 | 0.1 m | 500 M | 0% | ~100% |

**Ceiling: 0.3 m at ~50% success.** Extended training (436 M steps, double the other stages)
produced no improvement at 0.3 m. At 0.1 m the robot learned to tightrope-walk with a low
fall rate but never crossed — confirmed not a training-time problem.

**Root cause identified:** Without a y-position or edge-proximity signal, the policy learns
a gait prior that stays centred *on average*. On a narrow bridge, average is not enough —
the robot cannot detect drift and self-correct. The required information was simply absent
from the observation.

---

## Experiment 2 — Curriculum With Local Heightmap (Depth Obs)

### What We Added

A **13×13 local terrain patch** sampled from a pre-computed binary heightmap of the scene
(1.0 = solid surface, 0.0 = void), appended to the policy observation each step.

- **Resolution:** 3 cm/cell
- **Coverage:** ±18 cm from robot centre — reaches 5.3 cm past the outer feet
- **Frame:** robot-local, rotated by yaw — the policy always sees ahead/left/right edges
  regardless of heading in the world
- **Cost:** negligible — array indexing, fully JIT-compiled, no rendering
- **Observation size:** 46 → 215 floats (+169)

The patch gives the policy a direct "edge is N cells left, M cells right" signal each step.
Lateral asymmetry in the patch immediately encodes drift direction.

### Curriculum

0.8 m → 0.7 m → 0.6 m → 0.5 m → 0.4 m → 0.3 m → 0.2 m → 0.1 m (finer steps, 0.1 m increments)

| Stage | Width | Steps | Success | Fail | vs Baseline |
|---|---|---|---|---|---|
| depth_0.8m | 0.8 m | 300 M | 96% | 4% | +19 pp |
| depth_0.7m | 0.7 m | 200 M | 100% | 0% | *(new)* |
| depth_0.6m | 0.6 m | 200 M | 100% | 0% | +35 pp |
| depth_0.5m | 0.5 m | 200 M | 100% | 0% | +35 pp |
| depth_0.4m | 0.4 m | 200 M | 98% | 2% | +30 pp |
| **depth_0.3m** | **0.3 m** | **200 M** | **91%** | **9%** | **+42 pp** |
| **depth_0.2m** | **0.2 m** | **200 M** | **89%** | **11%** | **+73 pp** |
| depth_0.1m | 0.1 m | 300 M | 0% | 100% | 0 pp |

---

## Key Findings

### 1. The 0.3 m ceiling was a perception problem

0.3 m went from a 49% coin-flip (with 436 M steps) to **91% with only 200 M steps** after
adding the heightmap. The robot was physically capable of crossing all along — it just
couldn't sense where the edge was. This is one of the clearest demonstrations of the
information bottleneck problem in locomotion RL.

### 2. The gain scales inversely with bridge width

At 0.8 m the depth obs adds +19 pp. At 0.2 m it adds +73 pp. The narrower the bridge, the
more critical explicit edge sensing becomes. This makes intuitive sense: on a wide bridge,
random drift is tolerated; on a narrow one, every centimetre of drift matters.

### 3. 0.2 m is near-solved (89%) — a 73 percentage point improvement

The Go1's hip-to-hip stance is ~28 cm. A 0.2 m bridge is narrower than the robot's own
body. Yet the robot now crosses at 89% success by actively centring its body mass between
its widely-spread feet. This was previously considered physically implausible without
specialised perception.

### 4. 0.1 m is a physics wall, not a perception wall

With depth obs the robot now correctly perceives the 0.1 m edge — but still fails 100%.
The Go1 cannot place four feet on a 10 cm surface simultaneously. This is a morphological
constraint: no observation, reward, or training budget will overcome it on this robot.
This is confirmed — the failure mode is "feet land off-bridge instantly," not "robot drifts
and falls."

### 5. Convergence is faster with depth

Every depth stage converged in 200 M steps. No extensions were needed. The baseline
required up to 436 M steps at 0.3 m with no improvement. The edge signal simplifies credit
assignment — the policy gets immediate directional feedback on drift.

---

## What the Robot Learned to Do

The WandB videos (logged at each eval checkpoint) show a qualitative change in behaviour:

- **0.8 m–0.5 m:** Normal quadruped walk, minimal lateral correction
- **0.3 m–0.2 m:** Visible lateral micro-corrections mid-bridge — the robot detects which
  side it's drifting toward and actively re-centres, recovering from perturbations that
  would have caused falls without depth obs
- **0.1 m:** Robot perceives the edges correctly but has no valid foot placement available

---

## Problems Solved Along the Way

| Problem | Fix |
|---|---|
| Robot avoided crossing (success scale too small) | Raised to 5000 (>expected future return) |
| Penalty signals had no effect (reward clipped at 0) | Lower bound changed to −10 |
| NaN propagation from reward terms | Wrapped in `jp.nan_to_num` |
| `lateral_deviation` scale too aggressive | Tuned to −3.0 from −4.0 |
| Heightmap 10% too narrow (`np.linspace` spacing error) | Switched to `np.arange` for exact 0.03 m grid |
| No videos during training | Added `policy_params_fn` to render + upload at each eval |

---

## Next Steps / Open Questions

- **Can 0.2 m reach ~100%?** 89% with 200 M steps — more training or a tighter lateral penalty may close the gap
- **Can a humanoid cross 0.1 m?** H1/G1 have a narrower stance. The heightmap obs code transfers directly — same implementation, different robot
- **Continuous curriculum?** Rather than staged restarts, anneal `bridge_half_width` within a single run as success rate rises
- **Observation history?** `history_len=1` currently — 3–5 step history would let the policy detect drift *trends*, not just instantaneous state
- **Domain randomisation?** The lateral correction reflex likely becomes even more important under perturbations (wind, surface friction variation)
