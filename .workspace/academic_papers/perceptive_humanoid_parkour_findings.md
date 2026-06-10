# Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching

**Authors:** Zhen Wu, Xiaoyu Huang, Lujie Yang et al.
**Affiliation:** Amazon FAR / UC Berkeley / CMU / Stanford
**Year:** 2026 (arXiv:2602.15827v2, May 2026)

## Summary

This paper trains a humanoid robot to execute chained parkour skills (jumping, vaulting, climbing over high obstacles) by combining reference-motion tracking (DeepMimic-style) with PPO RL and a learned perceptive module. The core contribution is a motion-matching curriculum that chains short motion clips together, allowing the robot to learn dynamic whole-body manoeuvres that pure reward-shaped RL cannot discover. Zero-shot sim-to-real transfer is demonstrated on a Unitree H1 humanoid.

## Relevant Findings for the 0.1m Bridge Problem

- **Reward-shaping-only RL scores 0.00 on hard obstacles — a direct structural analog to our 0.1m wall.** Table I shows their RL-only baseline (identical observation quality to their full method, identical reward engineering effort) achieves 0.00 success rate on 58cm and 76cm obstacles at both tested speeds. The paper states explicitly: "the baseline relies on foot-only stepping and does not discover whole-body climbing strategies… highlighting the limitations of reward-shaping RL alone." This reframes our "physics wall" as potentially a **strategy-discovery wall** — PPO cannot stumble into a narrow-stance centerline gait via random exploration, just as their baseline cannot discover climbing without a motion prior.

- **Reference-motion tracking is what unlocks hard strategies (0.00 → 0.95–0.99).** Adding a DeepMimic-style tracking term that rewards Gaussian proximity to a reference trajectory on body position, orientation, and velocity (σ values reported in Table IV) takes success from 0.00 to 0.95–0.99 on the same hard obstacles. The motion prior shapes the exploration distribution into the target behavior basin — without it, the policy is stuck in a local optimum (standard walking) and never discovers the required skill.

- **The motion prior does not need to be physically perfect — only directionally correct.** The reference motions are hand-authored kinematic clips retargeted from human motion capture, not dynamically optimal trajectories. The RL component corrects for physical infeasibility. This means a hand-authored narrow-stance walking clip for the Go1 (tandem or near-tandem gait, feet close to centerline) could serve as the reference even if it is slightly unstable — PPO would stabilize it.

- **Hybrid DAgger + RL is essential for brief high-magnitude torques.** Pure per-step imitation loss underestimates peak efforts needed during dynamic transitions; adding a PPO alive/progress reward term alongside the imitation loss resolves this. For bridge crossing, a simple alive + forward-progress RL reward alongside imitation tracking is sufficient — no detailed reward engineering required.

- **Multi-skill chaining via motion matching at transition boundaries.** Skills are stitched together by matching body state at the end of one clip to the start of the next, enabling dynamic sequences without retraining. For our curriculum, this suggests treating each bridge-width stage as a "skill" with a known entry state (standard walk onto bridge) and exit state (walk off bridge) — potentially allowing fine-grained chaining of gaits at different widths.

- **Kinematic feasibility must be verified before attempting reference-motion bootstrap.** This paper cannot determine whether the Go1 can physically hold all four feet within a 10 cm lateral spread. The Go1's hip abductor range of motion limits how tightly feet can be placed. If the robot physically cannot adopt a narrow enough stance (i.e., hip adduction angle would exceed joint limits), no motion prior will help — this check must come first.

- **Perceptive module (heightmap + depth) enables skill triggering based on terrain proximity.** The paper uses terrain distance to trigger skill transitions — the robot begins a parkour move when the obstacle is within a learned threshold. For bridge crossing, an analogous trigger could initiate the narrow-stance gait when the heightmap detects bridge width below a threshold, transitioning back to normal gait on the far platform.

## Recommendation

First verify Go1 hip adduction limits: check whether the robot can physically hold all four feet within a ~5 cm lateral spread of the bridge centerline (i.e., ~2.5 cm each side) without exceeding joint limits. If feasible, add a DeepMimic-style reference-motion tracking reward seeded with a hand-authored narrow-stance walking clip — this paper's evidence (0.00 → 0.95+ on hard obstacles) strongly suggests reward-shaped RL alone cannot discover this strategy class, and a motion prior is the missing ingredient.
