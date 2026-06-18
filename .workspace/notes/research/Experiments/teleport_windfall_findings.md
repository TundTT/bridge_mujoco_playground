# Finding: Teleport Detection Removes Accidental `unpaid_progress` Windfall

> Observed 2026-06-12 during v6 curriculum launch.
> Comparison: v4_stage0 (commit 2657093) vs v6 Stage 0 (WandB `bridge_crossing_v6`).
> Status: **INCONCLUSIVE** — needs dedicated investigation.

---

## Background

`bridge.py` uses a contact-gated frontier reward: forward progress is banked in `unpaid_progress` and paid out at foot contact. In v4_stage0, the info dict was only reset on `done=True` (fall or success). Timeout episodes (robot walks but doesn't reach goal in time) triggered AutoReset, which teleported the robot back to spawn — but `unpaid_progress` was not cleared.

v6 introduced explicit teleport detection:

```python
teleported = state.data.qpos[0] > data.qpos[0] + 0.5
boundary = done | teleported
```

This resets `unpaid_progress` (and all info fields) on timeouts, preventing credit from carrying across episode boundaries.

---

## The Accidental Windfall Mechanism (v4 behavior)

1. Robot walks forward during a timeout episode, banking progress in `unpaid_progress`.
2. Episode times out → AutoReset teleports robot to spawn.
3. `unpaid_progress` is NOT cleared (only `done` triggers resets in v4).
4. Next episode begins at spawn. At first foot contact, the entire banked progress pays out as a large frontier reward — a windfall.
5. This windfall consistently occurs at the start of each post-timeout episode, creating a strong accidental learning signal that may encourage early foot contact behavior.

---

## Observed Effects

### Convergence speed

| Run | Success at 34M steps | Notes |
|---|---|---|
| v4_stage0 (0.8m) | ~88% | No teleport detection |
| v6 Stage 0 (0.8m) | ~30% | Teleport detection active |

v6 converges ~17M steps slower at the 0.8m stage. Both reach ~96%+ by 100M steps.

### Gait quality (visual, informal)

User observed v4 gait looks better in video at similar training stages — v4 shows careful trotting, v6 gait appears less deliberate. This is a subjective observation from a single comparison and has not been quantified.

Reference v4_stage0 final metrics (150M steps, 0.8m):
- `term_success`: 1.00
- `touchdown_count`: ~198/episode
- `abduction_saturation`: ~8.8%

v6 Stage 0 final metrics not yet available (run in progress as of writing).

---

## Hypothesis

The windfall at episode start may act as an accidental curriculum signal: the robot learns that foot contact immediately after spawn earns a large reward, reinforcing foot-contact habits early in training. This could produce more deliberate, contact-aware locomotion.

Alternatively, the windfall may simply accelerate initial convergence without meaningfully affecting final gait quality — the difference in gait quality may be explained by other variables (video sample timing, random seed, etc.).

---

## Open Questions

- Does v6 reach comparable final gait quality to v4 at 150M steps? (Awaiting v6 Stage 0 final eval)
- Is the gait quality difference consistent across multiple seeds, or a sampling artifact?
- Would reverting to `boundary = done` (removing teleport detection) reproduce v4 gait while keeping all other v6 fixes?
- Is the windfall effect beneficial on harder stages (narrower bridges), or only at 0.8m?

---

## Proposed Ablation

Run a controlled comparison at 0.8m / 150M steps, identical config, two variants:
- **A**: `boundary = done | teleported` (current v6 behavior)
- **B**: `boundary = done` (v4 behavior, no teleport detection)

Compare: `touchdown_count`, `abduction_saturation`, gait videos, `term_success` convergence curve.

**Do not revert teleport detection without running this ablation first.** The teleport detection was added to prevent reward leakage across episode boundaries; removing it is not a free win.
