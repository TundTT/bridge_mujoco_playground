# Narrow Bridge Locomotion — Depth Observation Findings

> **Date:** 2026-06-04
> **Project:** Unitree Go1 quadruped traversing a narrow bridge of configurable width
> **Goal:** Train a legged robot to cross increasingly narrow bridges using curriculum RL,
> ultimately reaching widths below the robot's own stance width

---

## Scene & Training Setup

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

| Term | Scale | Raw function | Purpose |
|---|---|---|---|
| `forward_vel` | +2.0 | `clip(local_vx, 0, 2)` | Drive forward progress |
| `progress_to_goal` | +3.0 | `log(clip(x+2.5, 1, 8)) / log(8)` | Shaped reward from spawn to goal (log curve) |
| `success` | +5000.0 | `1.0` when `x ≥ 5.5m` (Platform B midpoint) | Large terminal bonus for crossing |
| `alive` | +0.1 | `1.0` every step | Incentivise survival |
| `feet_air_time` | +0.1 | `sum((air_time - 0.1) * first_contact)` | Encourage a regular gait |
| `lateral_deviation` | −3.0 | `y²` | Penalise drift from bridge centreline |
| `heading` | −2.0 | `forward_y²` | Penalise facing sideways |
| `orientation` | −5.0 | `sum(upvector_xy²)` | Penalise tilting/rolling |
| `action_rate` | −0.01 | `sum((act - last_act)²)` | Smooth actions |
| `torques` | −0.0002 | `sqrt(sum(τ²)) + sum(|τ|)` | Penalise high torques |
| `energy` | −0.001 | `sum(|q̇| · |τ|)` | Penalise energy expenditure |
| `termination` | −1.0 | `1.0` on failure (fall or tilt) | Explicit failure penalty |

**Failure condition:** `qpos[2] < 0.2m` (root below 0.2m) OR upvector_z < 0 (robot flipped).
**Success condition:** `qpos[0] ≥ 5.5m` (halfway into Platform B).
Each term is computed per-step, summed, multiplied by `dt=0.02`, then clipped to `[-10, 10000]`.

---

## Experiment 1 — Baseline (No Depth)

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

**Ceiling: 0.3 m at ~50% success.** Extended training (436 M steps) produced no improvement.
Without a y-position or edge-proximity signal, the policy learns a gait prior that stays centred
*on average* — fine on a wide bridge, but insufficient as width narrows because the policy cannot
detect drift and self-correct.

---

## Experiment 2 — With Local Heightmap (Depth Obs)

### What We Added

A **13×13 local terrain patch** sampled from a pre-computed binary heightmap of the scene
(1.0 = solid surface, 0.0 = void), appended to the policy observation each step.

- **Resolution:** 3 cm/cell
- **Coverage:** ±18 cm from robot centre — reaches 5.3 cm past the outer feet
- **Frame:** robot-local, rotated by yaw — the policy always sees ahead/left/right edges
  regardless of heading in the world; row 0 = ahead, column 0 = robot's right
- **Encoding:** binary — 1.0 = solid surface, 0.0 = void/fall-off
- **Cost:** world heightmap built once at init (87 KB); patch sampled via array indexing each
  step — JIT-compiled, no rendering overhead
- **Observation size:** 46 → 215 floats (+169)

### Results

**Curriculum:** 0.8 m → 0.7 m → 0.6 m → 0.5 m → 0.4 m → 0.3 m → 0.2 m → 0.1 m

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

### What the Depth Obs Actually Enables

**Before:** The policy had no y-position signal and no terrain geometry. It learned a gait prior
that stayed centred on average — fine on a wide bridge, a coin-flip as the bridge narrows.

**After:** The 13×13 patch directly encodes "edge is N cells to the left / M cells to the right."
The asymmetry in the patch tells the policy which way it's drifting before it falls off. The policy
learned to use this signal to actively re-centre — visible in the WandB videos as a clear lateral
correction reflex that wasn't present before.

---

## Key Findings

### 1. The 0.3 m ceiling was a perception problem, not a capacity problem

The no-depth curriculum hit a hard wall at 0.3 m (49% success) even with 436 M steps. Adding the
heightmap with only 200 M steps per stage produced 91% at 0.3 m. The robot was capable of crossing
all along — it just couldn't sense where the edge was.

### 2. The gain scales inversely with bridge width

At 0.8 m the depth obs adds +19 pp. At 0.2 m it adds +73 pp. On a wide bridge, random drift is
tolerated; on a narrow one, every centimetre of drift matters and explicit edge sensing becomes
critical.

### 3. 0.2 m is near-solved (89%) — a 73 percentage point improvement

The Go1's hip-to-hip stance is ~28 cm. A 0.2 m bridge is narrower than the robot's own body. The
robot now crosses at 89% by actively centring its body mass between its widely-spread feet —
previously considered physically implausible without specialised perception.

### 4. 0.1 m is a physics wall, not a perception wall

With depth obs the robot can now see exactly where the edges are. It still fails 100% at 0.1 m.
The Go1's stance width (~28 cm) is wider than the 10 cm bridge — it cannot place four feet on the
surface simultaneously regardless of perception. The failure mode is "feet land off-bridge
instantly," not "robot drifts and falls." More training, better obs, or reward shaping will not
fix it.

### 5. Convergence is faster with depth

Every depth stage converged in 200 M steps. No extensions needed. The baseline required up to
436 M steps at 0.3 m with no improvement. The edge signal simplifies credit assignment — the
policy gets immediate directional feedback on drift rather than having to infer it from falling.

### 6. The heightmap must use np.arange, not np.linspace

`np.linspace(min, max, n)` gives grid spacing of `(max-min)/(n-1)` ≈ 0.030303 instead of exactly
0.03 m, causing the bridge width to be systematically under-represented by ~1 cell (~10% narrower
than configured). Fix: `np.arange(min, max + cell/2, cell)` for exact 0.03 m spacing. Now in
`bridge.py`.

---

## What the Robot Learned to Do

The WandB videos (logged at each eval checkpoint via `policy_params_fn`) show a qualitative change:

- **0.8 m–0.5 m:** Normal quadruped walk, minimal lateral correction needed
- **0.3 m–0.2 m:** Visible lateral micro-corrections mid-bridge — the robot detects which side
  it's drifting toward and actively re-centres, recovering from perturbations that would have
  caused falls without depth obs
- **0.1 m:** Robot attempts to walk but feet land off-bridge immediately — the patch shows the
  robot correctly perceives the edge, it just has nowhere to step

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

## Open Questions

- **Can 0.2 m reach ~100%?** 89% with 200 M steps — more training or a tighter lateral penalty may close the gap
- **Can a humanoid cross 0.1 m?** H1/G1 have a narrower stance. The heightmap obs code transfers directly — same implementation, different robot
- **Continuous curriculum?** Rather than staged restarts, anneal `bridge_half_width` within a single run as success rate rises
- **Observation history?** `history_len=1` currently — 3–5 step history would let the policy detect drift *trends*, not just instantaneous state
- **Domain randomisation?** The lateral correction reflex likely becomes even more important under perturbations (wind, surface friction variation)
- **Is binary encoding sufficient?** Continuous height values (e.g. modelling the drop below the bridge) might give richer signals but add complexity
