# Problems Encountered

A running log of issues hit during the bridge crossing curriculum project, what caused them, and how they were resolved.

---

## 1. Robot avoided crossing — success bonus too small

**Symptom:** The policy learned to hover near x≈4.0 (just before Platform B) indefinitely rather than crossing. Success rate stayed near 0% even when the robot was clearly capable of crossing.

**Cause:** The success bonus (initially scale=100, then 300) was smaller than the expected future return from continuing to survive. Hovering at x=4 for ~650 remaining steps earned ~75 reward units; a 2–6 unit one-time success bonus wasn't worth terminating for.

**Fix:** Raised success scale to 5000, giving a ~100-unit one-time bonus that exceeds expected future returns. Rule of thumb: `success_scale > mean_episode_reward / dt`.

---

## 2. Reward clip at 0 silenced all penalty signals

**Symptom:** Policy wasn't learning to avoid bad behaviours (tilting, lateral drift, high torques). Penalty terms were visible in logs but had no effect.

**Cause:** Reward was clipped to `[0, 10000]`. When penalty terms dominated positive rewards early in training, the total was always clipped to 0 — the gradient saw no difference between "slightly bad" and "very bad."

**Fix:** Changed lower bound to `-10.0` so the policy can distinguish degrees of failure.

---

## 3. NaN propagation from reward terms

**Symptom:** Training occasionally produced NaN rewards, destabilising the policy.

**Cause:** Certain reward terms (e.g. `feet_air_time`) could produce NaN under edge-case conditions, which then propagated through the sum.

**Fix:** Wrapped the total reward in `jp.nan_to_num(..., nan=0.0)` before clipping.

---

## 4. lateral_deviation scale too aggressive

**Symptom:** When `lateral_deviation` scale was raised to -4.0 to force tighter bridge alignment, fall rate rose from ~20% to ~30% and success rate learning slowed compared to -2.0.

**Cause:** Too strong a lateral penalty caused the robot to over-correct, destabilising its gait and increasing falls.

**Fix:** Tuned to -3.0, which gave the best balance: 78% success, 22% fall rate.

---

## 5. Robot has no bridge perception (core limitation)

**Symptom:** Performance plateaued at ~50% success on 0.3m bridge and collapsed completely on 0.2m and 0.1m. Extended training (300M+ steps) produced no improvement in success rate — only better fall avoidance.

**Cause:** The `state` observation contains no y-position and no edge-proximity signal. The robot can feel lateral velocity drift but has no explicit "am I centred?" feedback. By the time a foot loses contact with the bridge, it's already too late to recover.

**What the robot can do without perception:** Learn a gait prior that minimises lateral drift on average. This works on wide bridges (0.8m–0.4m) but becomes a coin flip at 0.3m and fails entirely below 0.2m.

**Fix (not yet applied):** Add y-position and foot-on-bridge contact signals to `state` in `bridge.py → _get_obs()`. Both are already computed — y-position is `data.qpos[1]`, foot contact is in `privileged_state` but not `state`.

---

## 6. 0.2m bridge narrower than robot stance

**Symptom:** Sudden cliff in curriculum performance between 0.3m (49% success) and 0.2m (16% success, never stable).

**Cause:** The Go1's stance width is ~0.28m. A 0.2m bridge means no configuration of the four feet can all be on the bridge simultaneously. The robot must balance with feet hanging off the edge.

**Note:** This isn't necessarily a hard blocker — the robot did achieve ~16% success occasionally — but without perception it can't do this reliably.

---

## 7. Grokking did not occur at 0.1m

**Symptom:** After 500M+ total steps at 0.1m bridge width, success stayed at exactly 0% despite fall rate dropping to ~2–5% and forward velocity continuing to improve.

**Cause:** Grokking requires the network to have access to the information needed to solve the task. The policy lacks y-position and edge sensing, so no amount of training time can produce a reliable crossing strategy — the required information simply isn't in the observation.

**Observation:** The robot did learn to tightrope-walk (low fall rate, high forward velocity) but timed out on every episode before reaching the goal. It's solving the wrong sub-problem.

---

## 8. WandB project move not available via SDK

**Symptom:** Attempted to move a run to a different WandB project programmatically; `run.move()` does not exist in the SDK.

**Fix:** Runs must be moved manually via the WandB web UI. The foundation run was renamed `foundation_1` as a workaround but remains in the original project.

---

## 9. NCCL/XLA multi-GPU contention (IOMMU P2P)

**Symptom:** Multi-GPU training showed degraded performance due to IOMMU blocking peer-to-peer GPU transfers.

**Fix:** Documented in `xla_jit_compilation_and_gpu_setup.md` — required BIOS/IOMMU configuration. See also commit `bc20a34`.

---

## 10. PPO step counter overshoots configured num_timesteps

**Symptom:** Runs configured for e.g. 150M steps consistently reported final step counts of ~236M in logs.

**Cause:** Brax PPO step quantization — actual steps = `num_evals × ceil(total / num_evals / chunk_size) × chunk_size` where `chunk_size = num_envs × unroll_length = 81920`. The ceiling rounds each eval interval up, compounding across all evals.

**Impact:** Cosmetic only — training runs longer than specified but behaviour is correct. Step counts in checkpoints and WandB reflect actual steps trained.
