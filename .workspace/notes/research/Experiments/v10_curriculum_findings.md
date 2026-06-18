# v10 Curriculum: Full Target Width Achieved — 0.10m at 91.4% (need rewrite for lab (assume people reading this have no prior knowlage of the project))

> Run completed 2026-06-13. WandB project: `bridge_crossing_v10`, group: `v10_curriculum`.
> Script: `run_curriculum_v10.sh`. Status: **all 8 stages complete.**
> **This is the first run to reach the 0.10m physical target.** Previous best: v6 at 0.16m (92%).

---

## What Changed vs v6

v6 reached 0.16m at 92% but hit a hard ceiling there. Four categories of fix were required to break through:

### 1. World model correctness (bugs, not reward design)

**Symmetric y-grid** (`_build_world_heightmap`): the old `np.arange(-1.0, 1.0, 0.03)` grid never contained y=0, shifting the perceived bridge laterally by ~1.5cm. The policy trained on an asymmetric world model and anchored its stance to the wrong physical edges. Fixed to integer multiples of cell size centred on 0:
```python
n_half = int(round((_HM_Y_MAX - _HM_Y_MIN) / 2 / _HM_CELL))
ys = np.arange(-n_half, n_half + 1) * _HM_CELL
```

**Analytic `foot_off_bridge`**: the heightmap-lookup version had 3cm cell quantisation introducing ~1cm boundary error. Replaced with direct `|foot_y| <= hw` check — exact, no quantisation.

### 2. Dead-gradient fix (the key breakthrough)

The old quality function:
```python
quality = jp.clip(margin / jp.minimum(info["virtual_hw"], 0.10), 0.0, 1.0)
```
saturated at 0 once a foot was outside `virtual_hw`. At natural stance (|y|~0.14m) against `vhw=0.05`, all feet had `quality=0` and **the frontier term had zero inboard gradient**. The fading floor and `foot_off_virtual` were doing all the narrowing work alone — and they were barely enough. This is why v6 stalled at 0.16m.

Fix: soft exponential quality with 5cm decay length:
```python
quality = jp.exp(jp.minimum(margin, 0.0) / 0.05)
```
This gives `quality=1.0` inside `virtual_hw` and decays smoothly outside, providing a continuous inboard gradient from any stance width. At natural stance vs vhw=0.05: quality≈0.165, giving meaningful frontier signal even from far outside.

### 3. Reward exploit fixes (v8/v9 work)

Three gait exploits were identified and closed before v10 launched:

**Contact-only quality_mean → all-feet quality_mean**: the original `foothold_multiplier` averaged quality over *contacting* feet only. Lifting a low-quality foot *raised* the multiplier, making 2-legged "training wheels" gait profitable. Fixed to `jp.mean(quality)` over all 4 feet unconditionally.

**Contact-independent `foot_off_virtual`** (scale −10): originally contact-gated (fired only on contact steps), which meant bounding — fewer contact events — paid the penalty less. Now fires on all feet every step.

**`feet_stale_air`** (scale −2.0, threshold 0.4s): penalises legs held airborne >0.4s. Killed the pattern where the robot used front-left + back-left as primary stance while hovering the right side as stabilisers.

**Pose abduction weight 1.0 → 0.1**: removed the adduction tax that competed with the narrow-stance signal at tight virtual widths.

### 4. Curriculum structure fixes

**Midpoint insertion at 0.20m**: the first attempt at stage 4 (0.16m) peaked at 75.8% then collapsed to 0% — the 0.24m→0.16m jump was too steep. A 0.20m intermediate stage was inserted. Both 0.20m and 0.16m then passed at 100%.

**`vhw_min` below `hw` at the narrowest stages**: stages 6 and 7 use `vhw_min` tighter than the physical edge so `foot_off_virtual` and the frontier gradient remain active even when feet sit exactly at the physical boundary.

---

## Final Reward Scales

| Term | Scale | Notes |
|---|---|---|
| `frontier_delta` | +50.0 | Contact-gated, weighted by `foothold_multiplier` |
| `feet_air_time` | +2.0 | Cap 0.15s |
| `success` | +5000.0 | |
| `termination` | −200.0 | |
| `foot_off_bridge` | −50.0 | Analytic, exact boundaries |
| `foot_off_virtual` | −10.0 | All feet, every step, contact-independent |
| `feet_stale_air` | −2.0 | Linear above 0.4s |
| `orientation` | −5.0 | |
| `lateral_deviation` | −3.0 | |
| `heading` | −2.0 | |
| `lin_vel_z` | −2.0 | |
| `pose` | +0.5 | Abduction weight 0.1, hip/knee 1.0 |
| `feet_clearance` | −2.0 | |
| `feet_slip` | −0.25 | |
| `dof_pos_limits` | −1.0 | |
| `action_rate` | −0.01 | |
| `torques` | −0.0002 | |
| `energy` | −0.001 | |
| `feet_height` | −0.2 | |

`foothold_multiplier`:
```python
quality = jp.exp(jp.minimum(margin, 0.0) / 0.05)       # soft, always live
quality_mean = jp.mean(quality)                          # all 4 feet
floor = 0.5 * jp.clip((vhw - 0.10) / 0.05, 0.0, 1.0)   # fades at narrow
foothold_multiplier = floor + (1.0 - floor) * quality_mean
```

---

## Curriculum Structure

8 stages (0.20m midpoint inserted after stage 4 failure on first v10 attempt):

| Stage | Width | hw | vhw_min | Budget | ep_len |
|---|---|---|---|---|---|
| 0 | 0.8m | 0.40 | 0.15 | 150M | 1000 |
| 1 | 0.4m | 0.20 | 0.10 | 100M | 1000 |
| 2 | 0.32m | 0.16 | 0.08 | 100M | 1000 |
| 3 | 0.24m | 0.12 | 0.06 | 150M | 1000 |
| 4 | 0.20m | 0.10 | 0.05 | 150M | 1500 |
| 5 | 0.16m | 0.08 | 0.05 | 200M | 1500 |
| 6 | 0.13m | 0.065 | 0.045 | 150M | 1500 |
| 7 | 0.10m | 0.05 | 0.040 | 200M | 1500 |

---

## Results

All 8 stages passed the gate first attempt (after midpoint insertion):

| Stage | Width | Success | TDs/ep | Contact L/R | max_foot_y (sum) | Gate |
|---|---|---|---|---|---|---|
| 0 | 0.8m | **100%** | 91 | — | 76.1 | ✅ Pass |
| 1 | 0.4m | **100%** | 83 | — | 53.7 | ✅ Pass |
| 2 | 0.32m | **100%** | 157 | — | 43.1 | ✅ Pass |
| 3 | 0.24m | **99.2%** | 114 | — | 29.2 | ✅ Pass |
| 4 | 0.20m | **100%** | 130 | 312 / 230 | 28.5 | ✅ Pass |
| 5 | 0.16m | **100%** | 175 | 283 / 223 | 28.3 | ✅ Pass |
| 6 | 0.13m | **98.4%** | 131 | 313 / 224 | 21.6 | ✅ Pass |
| 7 | 0.10m | **91.4%** | 171 | 274 / 202 | 23.3 | ✅ Pass |

**Final checkpoint**: `logs/Go1BridgeCrossing-20260612-203650/checkpoints/000200540160`

### Contact symmetry

Left/right contact is now balanced throughout (L/R ratio ~1.3–1.4x, vs 2.8x asymmetry in the v8 training-wheels run). The `feet_stale_air` penalty and contact-independent `foot_off_virtual` successfully prevented lateral gait asymmetry.

### max_foot_y progression

Summed over the episode, `max_foot_y` decreases from 76 (0.8m) to 21–23 (0.10m), consistent with the robot genuinely narrowing its stance across the curriculum. The policy is not memorising a fixed stance — it's tracking `virtual_hw`.

---

## What Didn't Work Along the Way

**v8 — contact-gated `foot_off_virtual`**: firing only on contact steps created a cadence incentive. Bounding (fewer contacts) paid the penalty less. Switched to contact-independent.

**v8 — contact-only `quality_mean`**: lifting a low-quality foot raised the multiplier. Led to the "training wheels" 2-legged gait. Switched to all-feet mean.

**v9 — hard clip quality**: fixed the exploit but left a dead gradient outside `virtual_hw`. The frontier term gave zero inboard pull once feet were outside the virtual boundary. This was the ceiling that blocked v6 and v9. Switched to soft exponential.

**v10 first attempt — 0.24m → 0.16m jump**: stage 4 peaked at 75.8% then catastrophically collapsed to 0% at the end of training. Contact log showed right-biased gait re-emerging just before collapse. The 0.20m midpoint fixed this.

---

## Open Questions

- **Physical hardware transfer**: the policy was trained at physical hw=0.05 (0.10m bridge). Does it zero-shot to a real 0.13m bridge? The physical target is hw=0.065, which was crossed at 98.4% in simulation.
- **Contact asymmetry root cause**: L/R contact ratio is ~1.35 consistently across all stages. The world model is now symmetric, so this is likely a gait preference (diagonal left-leading trot) rather than a world-model artifact. Worth investigating if sim-to-real transfer shows lateral drift.
- **Can 0.10m improve further?** 91.4% is strong but not 100%. Extensions (additional 100M steps) were not needed to pass the gate — but if physical transfer degrades, more training at 0.10m may help.
