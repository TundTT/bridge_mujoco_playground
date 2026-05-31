# Potential Improvements for Next Foundation Run

This captures things we learned from the current curriculum run that we'd do differently
when starting a fresh foundation. None of these require restarting now — save for next time.

---

## 1. The robot has no bridge perception (most important)

**Problem:** The policy `state` observation has no y-position and no explicit bridge-edge
signal. The robot cannot sense how close it is to the edge — only that it has *already fallen
off* (foot contact loss). At 0.3m width this becomes a coin flip.

**What to add to `state` in `bridge.py` → `_get_obs()`:**

```python
# 1. y-position — tells the policy how far off-centre it is
jp.array([data.qpos[1]])                    # 1 value

# 2. foot-on-bridge contact — tells the policy which feet are still on the bridge
# (already computed as `contact`, shape (4,))
# currently only in privileged_state, not state
```

The `privileged_state` already has this; it just needs to be promoted into `state` so the
deployed policy (which only sees `state`) can use it.

**Why this matters:** Without these signals the policy learns a gait *prior* that stays
centred on average, which degrades to ~50/50 at narrow widths. With them it can learn a
reactive centering reflex — detect drift early and correct before falling off.

---

## 2. Reward clip lower bound

**Problem:** We started with `clip(reward, 0, 10000)` which silenced all penalty signals
when penalties dominated. Fixed mid-run to `clip(reward, -10, 10000)` but the foundation
was already trained without it.

**Fix:** Use `-10.0` as the lower bound from day one.

---

## 3. Success scale sizing

**Problem:** Started with `success=100`, then `300`, before landing on `5000`. The robot
actively avoided termination (hovering at x≈4 was more profitable than crossing) until
the bonus exceeded expected future returns (~75 reward units per episode).

**Rule of thumb for next time:** Set `success_scale > mean_episode_reward × dt × episode_length`.
With reward≈25, dt=0.02, episode_length=1000: minimum scale ≈ 500. We used 5000 (10×
margin) which worked well.

---

## 4. Curriculum staging

**What we did:** 0.8m → 0.5m → 0.4m → 0.3m → 0.2m, each ~150–300M steps.

**What to try:** Finer steps near the hard end (e.g. 0.35m, 0.30m, 0.25m) rather than
jumping 0.1m at a time. The biggest performance drop was 0.4m→0.3m (from ~68% to ~38%
warm start) suggesting the gap was too large.

Alternatively: **continuous curriculum within a single run** — start `bridge_half_width`
at 0.8 and anneal it as success rate rises, rather than staged restarts.

---

## 5. Observation history

**Current:** `history_len=1` — policy only sees the current timestep.

**Improvement:** Increase to 3–5 timesteps. A short history lets the policy detect trends
(am I drifting? am I oscillating?) rather than reacting only to instantaneous state.
Especially useful for edge-avoidance where the dangerous state is approaching-the-edge,
not already-at-the-edge.

---

## 6. Lateral deviation penalty shape

**Current:** `jp.square(y_pos)` — quadratic, so small deviations are under-penalised.

**Alternative:** Exponential penalty that grows sharply near the bridge edge:
```python
# Ramps steeply as y approaches bridge_half_width
def _cost_lateral_deviation(self, y_pos, half_width):
    normalized = jp.abs(y_pos) / half_width
    return jp.exp(3.0 * normalized) - 1.0
```
This would teach the robot to care more about being near the edge, not just being off-centre.
