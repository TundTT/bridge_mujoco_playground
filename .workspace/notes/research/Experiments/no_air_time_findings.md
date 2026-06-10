# Findings: No Feet-Air-Time Ablation

> Experiment completed 2026-06-05. WandB project: `bridge_crossing-depth-no-air`.
> Baseline comparison: Experiment 2 (`bridge_crossing_with_depth`), identical except `feet_air_time` reward term set to scale 0.0.
> Script: `run_curriculum_no_air.sh`. All 8 stages ran and completed in a single chained execution.

---

## What We Changed

Removed the `feet_air_time` reward term by setting its scale to 0.0. Everything else — heightmap obs, reward terms, PPO config, curriculum stages — was identical to Experiment 2.

The `feet_air_time` term rewarded regular gait rhythm: `sum((air_time - 0.1) * first_contact)` at scale +0.1. Its purpose was to encourage a trot rather than a shuffle or drag. The question was whether this term helps or hurts on narrow bridges, where the robot may need to deviate from a standard trot to maintain balance.

---

## Results

| Stage | Width | Exp 2 (with air time) | Ablation (no air time) | Change |
|---|---|---|---|---|
| stage1_0.8m | 0.8m | 96% | 89.1% | −7 pp |
| stage2_0.7m | 0.7m | 100% | 96.1% | −4 pp |
| stage3_0.6m | 0.6m | 100% | 96.1% | −4 pp |
| stage4_0.5m | 0.5m | 100% | 98.4% | −2 pp |
| stage5_0.4m | 0.4m | 98% | 95.3% | −3 pp |
| **stage6_0.3m** | **0.3m** | **91%** | **96.9%** | **+6 pp** |
| **stage7_0.2m** | **0.2m** | **89%** | **92.2%** | **+3 pp** |
| stage8_0.1m | 0.1m | 0% | 0% | 0 pp |

---

## What We Learned

### 1. `feet_air_time` trades wide-bridge performance for narrow-bridge performance

Removing the gait rhythm reward makes the robot slightly worse on wide bridges (−7 pp at 0.8m, −2 to −4 pp at 0.5m–0.7m) but measurably better on the hardest stages (+6 pp at 0.3m, +3 pp at 0.2m). The tradeoff is consistent across the curriculum.

The interpretation: on a wide bridge, an encouraged trot is fine — the robot has room to use a standard gait. On a narrow bridge, the trot rhythm may conflict with lateral correction; the robot has to choose between staying in gait phase and making a lateral adjustment. Without the gait incentive, it's freer to deviate from the trot when balance requires it.

### 2. 0.3m now reaches 97% — the strongest result yet

The combination of depth obs + no gait constraint pushes 0.3m success to 96.9%, compared to 91% with depth alone and 49% without depth. This is now close to the ceiling for this width given the current reward and observation setup.

### 3. 0.2m reaches 92% — small but consistent improvement

The 0.2m stage (bridge narrower than the robot's own stance) improved from 89% to 92.2%. The gain is modest but consistent with the 0.3m result.

### 4. 0.1m remains a physics wall

No change at 0.1m. This confirms the 0.1m failure is morphological regardless of reward shaping.

### 5. Wide-bridge degradation is acceptable

The −7 pp drop at 0.8m (from 96% to 89%) sounds significant but 89% is still high absolute performance. In a curriculum context, the wide-bridge stages are stepping stones, not the final goal. The tradeoff is worth it.

---

## Reward Function

Same as Experiment 2 except `feet_air_time` scale set to 0.0:

| Term | Scale | Change from Exp 2 |
|---|---|---|
| `forward_vel` | +2.0 | unchanged |
| `progress_to_goal` | +3.0 | unchanged |
| `success` | +5000.0 | unchanged |
| `alive` | +0.1 | unchanged |
| `feet_air_time` | **0.0** | **disabled** |
| `lateral_deviation` | −3.0 | unchanged |
| `heading` | −2.0 | unchanged |
| `orientation` | −5.0 | unchanged |
| `action_rate` | −0.01 | unchanged |
| `torques` | −0.0002 | unchanged |
| `energy` | −0.001 | unchanged |
| `termination` | −1.0 | unchanged |

---

## Open Questions

- **Is the gait quality worse without air time?** The robot may adopt an unusual gait (shuffle, wide stance, creeping) that works on the bridge but wouldn't transfer to uneven terrain. Worth checking the WandB videos for gait style at narrow widths.
- **Is there a middle ground?** A lower scale (e.g. +0.02 instead of +0.1) might preserve most of the gait benefit while reducing the conflict with lateral correction on narrow bridges.
- **Best curriculum going forward:** no-air-time ablation is now the strongest Go1 result. If starting a fresh foundation, disable `feet_air_time` from day one.
