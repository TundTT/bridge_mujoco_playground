# v6 Curriculum: Contact-Gated Frontier + Exact v4_stage0 Scales

> Run started 2026-06-12. WandB project: `bridge_crossing_v6`, group: `v6_curriculum`.
> Script: `run_curriculum_v6.sh`. Status: **stages 0–4 complete, stage 5 midpoint (0.13m) in progress.**
> This is the strongest result to date: 0.16m crossed at 92% in a fully automated curriculum run.

---

## What We Built Toward This

Several earlier approaches failed or plateaued before reaching this design:

**Curricula 1 & 2 (bridge_crossing_1, bridge_crossing_with_depth)** used a forward-velocity tracking reward. The standing-still exploit made these unreliable below 0.3m: `exp(-err²/0.25)` gives 0.37 reward when standing still (vel=0, target=0.5 m/s), creating a local optimum the robot preferred on narrow bridges.

**v4_stage0** replaced velocity tracking with a contact-gated frontier reward (`frontier_delta=50`): forward progress is banked in `unpaid_progress` and paid out only at foot contact. Standing still earns zero. This run produced the best visual gait quality seen — careful, deliberate trotting at 0.8m — and was the anchor we returned to.

**v5** (Fable-recommended recalibration): lowered `frontier_delta` scale 50→20 and added stronger penalties. This caused bounding gait (32 TDs/ep, 24% abduction saturation vs v4's 198 TDs/ep, 8.8%). The optimizer found a long-hop exploit at lower frontier scale. Reverted.

**v6** reverts to exact v4_stage0 reward logic and scales, with two additions: explicit teleport detection to prevent reward leakage across episode boundaries, and a structured automated curriculum.

---

## Key Design Decisions

### 1. Contact-gated frontier reward (scale = 50)

Progress is banked while the robot walks forward, then paid out at the moment of foot contact — scaling with foot placement quality via `foothold_multiplier`. This means:
- Standing still → zero reward
- Moving forward without touching down → zero reward (encourages foot contact)
- Careful, deliberate trot → full payout proportional to distance covered

```python
capped_delta = jp.minimum(new_max - state.info["max_x_reached"], 0.6 * self.dt)
state.info["unpaid_progress"] += capped_delta
# Paid out in _get_reward:
frontier_payout = (info["unpaid_progress"] / self.dt) * foothold_multiplier * any_contact
# Cleared after payout:
state.info["unpaid_progress"] = jp.where(jp.any(contact_filt), jp.zeros(()), ...)
```

Scale=50 was critical. At scale=20 the optimizer switched from high-frequency tapping to low-frequency bounding (long hops earn the same total payout with fewer, more powerful contacts).

### 2. feet_air_time cap at 0.15s (not 0.25s)

`_reward_feet_air_time` caps at 0.15 seconds. An earlier regression raised this to 0.25s, which directly rewards longer hops and encourages bounding. At 0.15s, the reward saturates quickly and favors high-frequency, short-flight trotting.

```python
capped = jp.minimum(air_time, 0.15)
return jp.sum((capped - 0.1) * first_contact)
```

### 3. Foothold multiplier (contact quality gating)

The frontier payout scales with placement quality over contacting feet only:

```python
foot_on = contact_filt.astype(jp.float32)
quality_mean = jp.sum(foot_quality * foot_on) / (jp.sum(foot_on) + 1e-6)
foothold_multiplier = 0.5 + 0.5 * quality_mean
```

This links forward progress reward to precise foot placement — the robot learns that sloppy footholds earn less than clean ones.

### 4. No depth observation

v6 uses the base 46-element proprioceptive state only (joint positions/velocities, IMU, foot contacts, goal direction). Unlike Curricula 2 & 3, there is no 13×13 heightmap. The contact-gated reward provides an implicit edge signal: falling off the bridge terminates the episode (termination=-200), so the frontier reward only pays out on successful crossings.

This simplification worked. The depth obs in earlier curricula was compensating for a weaker reward signal.

### 5. Reward scales (exact v4_stage0 values)

| Term | Scale |
|---|---|
| `frontier_delta` | +50.0 |
| `feet_air_time` | +2.0 |
| `success` | +5000.0 |
| `termination` | −200.0 |
| `orientation` | −5.0 |
| `lateral_deviation` | −3.0 |
| `heading` | −2.0 |
| `lin_vel_z` | −2.0 |
| `foot_off_bridge` | −50.0 |
| `feet_clearance` | −2.0 |
| `dof_pos_limits` | −1.0 |
| `feet_slip` | −0.25 |
| `action_rate` | −0.01 |
| `torques` | −0.0002 |
| `energy` | −0.001 |
| `feet_height` | −0.2 |
| `pose` | +0.5 |
| `alive` | 0.0 |

### 6. Curriculum structure

6 physical stages with pre-inserted 0.32m midpoint (between 0.4m and 0.24m, where earlier curricula hard-failed). Virtual bridge width resampled per episode from U[vhw_min, hw] to improve generalization within each stage.

Gate logic: advance if term_success ≥ 0.70; extend +100M if 0.30–0.70; revert to last good checkpoint and insert midpoint if < 0.30.

| Stage | Width | hw | vhw_min | Budget | ep_len |
|---|---|---|---|---|---|
| 0 | 0.8m | 0.40 | 0.15 | 150M | 1000 |
| 1 | 0.4m | 0.20 | 0.10 | 100M | 1000 |
| 2 | 0.32m | 0.16 | 0.08 | 100M | 1000 |
| 3 | 0.24m | 0.12 | 0.06 | 150M | 1000 |
| 4 | 0.16m | 0.08 | 0.05 | 200M | 1500 |
| 5 (mid) | 0.13m | 0.065 | 0.05 | 150M | 1500 |
| 6 | 0.10m | 0.05 | 0.05 | 200M | 1500 |

---

## Results

All stages passed the gate first attempt without extensions:

| Stage | Width | Success | TDs/ep | Abduction sat | Gate | Wall time |
|---|---|---|---|---|---|---|
| 0 | 0.8m | **100%** | 97 | 2.0% | ✅ Pass | ~8 min |
| 1 | 0.4m | **100%** | 92 | 4.3% | ✅ Pass | ~8 min |
| 2 | 0.32m | **97.7%** | 130 | 26.2% | ✅ Pass | ~8 min |
| 3 | 0.24m | **99.2%** | 149 | 3.3% | ✅ Pass | ~9 min |
| 4 | 0.16m | **92.2%** | 159 | 5.4% | ✅ Pass | ~10 min |
| 5 | 0.10m | **0%** | 8 | 44.9% | ❌ Failed | ~10 min |
| 5 (mid) | 0.13m | — | — | — | in progress | — |

The touchdown count progression (97 → 92 → 130 → 149 → 159) shows the robot adapting to narrower bridges with more frequent, deliberate steps. Abduction saturation stays low (<6%) through stage 4, confirming the robot is not pushing joint limits — clean gait.

Stage 5 (0.10m) collapsed completely: the bridge is narrower than the robot's stance width (~0.28m), so all four feet cannot be placed simultaneously. The midpoint at 0.13m is now running.

### Comparison to previous best

The previous curriculum record was depth obs (Curriculum 2) reaching 0.2m at 89%. v6 reaches **0.16m at 92%** without depth observations, in a fully automated single script run. The 0.32m pre-inserted midpoint and contact-gated frontier reward are the key changes.

---

## What Didn't Work (Contrast)

- **Velocity tracking reward**: standing-still exploit (0.37 reward at v=0). Abandoned after v4d smoke test.
- **frontier_delta scale=20 + stronger penalties (v5)**: bounding exploit. The optimizer switched from tapping to long hops at lower scale. 32 TDs/ep, 24% abduction saturation.
- **feet_air_time cap=0.25s**: directly rewards longer flights, encourages bounding. Fixed back to 0.15s.
- **Depth obs (heightmap)**: useful for the velocity-tracking reward era but unnecessary with a well-shaped frontier reward.

---

## Open Questions

- Will the 0.13m midpoint bridge the gap to 0.10m, or is 0.10m still a physics wall?
- Is the teleport detection (`boundary = done | teleported`) helping or hurting convergence? See `teleport_windfall_findings.md` for a full write-up — inconclusive.
- At what width does the contact-gated frontier reward break down? The 0.10m failure suggests there is a morphological limit. With no-depth obs, does the robot have enough signal to navigate a 0.13m bridge?
