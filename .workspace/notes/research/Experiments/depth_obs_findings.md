# Findings: Local Heightmap (Depth) Observation

> Experiment completed 2026-06-04. WandB project: `bridge_crossing_with_depth`.
> Baseline comparison: `bridge_crossing_1` (same curriculum, no depth obs).

---

## What We Added

A **13×13 local terrain patch** sampled from a pre-computed world heightmap and appended to the policy's `state` observation vector. The patch is always in robot-local frame — row 0 = ahead, column 0 = robot's right — so the policy sees the same spatial layout regardless of heading.

- **Grid resolution:** 3 cm/cell
- **Patch coverage:** ±18 cm from robot centre (5.3 cm past the outer feet)
- **Encoding:** binary — 1.0 = solid surface, 0.0 = void/fall-off
- **Observation size:** 46 → 215 floats (169 new)
- **Cost:** world heightmap built once at init (87 KB); patch sampled via array indexing each step — JIT-compiled, no rendering overhead

---

## Results

| Bridge width | No depth (success) | With depth (success) | Change |
|---|---|---|---|
| 0.8m | 77% | 96% | +19 pp |
| 0.6m | 65% | 100% | +35 pp |
| 0.5m | 65% | 100% | +35 pp |
| 0.4m | 68% | 98% | +30 pp |
| 0.3m | **49%** | **91%** | **+42 pp** |
| 0.2m | **16%** | **89%** | **+73 pp** |
| 0.1m | 0% | 0% | 0 pp |

The biggest gains are exactly where we expected: the widths where the robot's feet are close to or past the bridge edge. At 0.2m (bridge narrower than the robot's stance) the improvement is +73 percentage points.

---

## What the Depth Obs Actually Enables

**Before (no depth):** The policy had no y-position signal and no terrain geometry. It learned a gait prior that stayed centred on average — which is fine on a wide bridge but degrades to a coin-flip as the bridge narrows, because the policy cannot detect drift and has no reason to correct laterally.

**After (with depth):** The 13×13 patch directly encodes "edge is N cells to the left / M cells to the right." The asymmetry in the patch tells the policy which way it's drifting before it falls off. The policy learned to use this signal to actively re-centre — visible in the WandB videos as a clear lateral correction reflex that wasn't present before.

---

## What We Learned

### 1. The old ceiling was a perception problem, not a capacity problem

The no-depth curriculum hit a hard wall at 0.3m (49% success) even with 436M steps and multiple extensions. Adding the heightmap obs with only 200M steps per stage produced 91% at 0.3m. The robot was capable of crossing; it just couldn't tell where the edge was.

### 2. The depth obs pays off most below 0.4m

At 0.8m the gain is modest (+19 pp) because the robot could stay centred without explicit edge sensing — the bridge is wide enough that random drift rarely causes a fall. Below 0.4m the gain grows sharply, peaking at 0.2m (+73 pp) where the bridge is narrower than the robot's own stance and perfect centring is required.

### 3. 0.1m is a physics wall, not a perception wall

With depth obs the robot can now see exactly where the edges are. It still fails 100% at 0.1m. The reason: the Go1's stance width (~28 cm hip-to-hip) is wider than the 10 cm bridge. The robot cannot place four feet on the surface simultaneously regardless of what it perceives. This confirms the 0.1m failure is a morphology constraint, not an information problem. More training steps, better obs, or reward shaping will not fix it.

### 4. The heightmap must use np.arange, not np.linspace

During Stage 1 testing we found that using `np.linspace(min, max, n)` gives a grid spacing of `(max-min)/(n-1)` ≈ 0.030303 instead of exactly 0.03 m. This causes the bridge width to be systematically under-represented by ~1 cell (~10% narrower than configured). The fix is `np.arange(min, max + cell/2, cell)` which gives exact 0.03 m spacing. This is now in `bridge.py`.

### 5. Convergence is faster with depth

Every stage with depth obs converged to its final performance in 200M steps. The equivalent no-depth stages often required extensions (up to 436M total for 0.3m). The depth signal appears to simplify the credit assignment problem — the policy gets immediate feedback on drift direction rather than having to infer it from falling.

### 6. Videos reveal the qualitative change

The per-eval WandB videos (logged via `policy_params_fn` at each checkpoint) clearly show:
- At 0.8m: robot walks normally, little lateral correction needed
- At 0.3m–0.2m: robot makes visible lateral micro-corrections mid-bridge, using the edge signal to re-centre after perturbations
- At 0.1m: robot attempts to walk but feet land off-bridge immediately — the patch shows the robot correctly perceives the edge, it just has nowhere to step

---

## Reward Function

All runs used the same reward configuration (no changes from baseline curriculum).
Each term is computed per-step, summed, multiplied by `dt=0.02`, then clipped to `[-10, 10000]`.

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

The `lateral_deviation` and `heading` terms were the critical ones — they gave the robot a weak centering incentive even without depth obs. With depth obs, these terms combined with the explicit edge signal to produce the strong correction reflex.

---

## Open Questions

- **Can 0.2m reach ~100%?** It ended at 89% with 200M steps. More training or a lower `lateral_deviation` penalty threshold might push it to 95%+.
- **Can a humanoid (H1/G1) cross 0.1m?** Humanoids have a narrower effective stance and different foot geometry. The heightmap obs transfers directly — same code, different robot.
- **Does the heightmap help with dynamic perturbations?** We only tested the static bridge. Under domain randomisation (wind, slippery surface) the lateral correction reflex may be even more important.
- **Is binary encoding sufficient?** We used 1.0/0.0. Continuous height values (e.g. modelling the drop below the bridge) might give richer signals but add complexity.
