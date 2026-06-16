# Narrow Bridge Crossing: From 0.8m to the Morphological Limit

> **Lab meeting document — 2026-06-16**
> WandB: `bridge_crossing_final` (presentation run, live)
> Codebase: `bridge_mujoco_playground` fork of Google DeepMind's `mujoco_playground`

---

## The Task

We train a **Unitree Go1 quadruped** to cross a narrow bridge using reinforcement learning in simulation (MuJoCo MJX + Brax PPO).

The environment consists of three regions:
- A wide **starting platform** (3m long, ~2m wide)
- A narrow **bridge** (4m long, configurable width)
- A wide **landing platform** (3m long, ~2m wide)

The robot spawns at the start, must traverse the bridge, and reach the far platform. Falling off terminates the episode with a large negative reward. The bridge width is the key variable — we systematically shrink it across a curriculum.

**Why this task?** Narrow bridge crossing requires the robot to do something genuinely hard: narrow its natural ~28cm stance to fit on bridges far smaller than its body width, while maintaining dynamic stability. It's a clean probe of locomotion capability and reward design quality, and the results transfer directly to real-world constrained-footing scenarios (stepping stones, beams, uneven terrain with edges).

**Robot morphology relevant to the task:**
- Natural stance width: ~28cm (feet at ±14cm from centerline)
- Foot collision sphere radius: 2.3cm
- Hip-to-hip width: ~18cm
- Abduction joints can bring feet well inboard of the hips

---

## Environment Setup

**Simulation**: MuJoCo MJX (GPU-accelerated), 4096 parallel environments, PPO (Brax implementation), control frequency 50Hz (dt=0.02s), γ=0.997, entropy cost=0.01.

**Observation** (46 elements, proprioceptive only — no depth camera):
- Joint positions and velocities (12 + 12)
- IMU: base linear velocity, angular velocity, orientation (3 + 3 + 3)
- Foot contact booleans (4)
- Goal direction vector (2)
- Virtual bridge half-width, normalized (1)
- Previous action (12)

**No depth or heightmap observation.** This was a deliberate design choice — earlier experiments showed that depth observations were compensating for weak reward signals. Once the reward was well-shaped, proprioception alone was sufficient.

**Virtual bridge width (vhw)**: A key design element. Each episode resamples a *virtual* bridge half-width from U[vhw_min, physical_hw]. Rewards and quality signals use the virtual width; physics uses the physical width. This creates a within-stage curriculum: some episodes are harder than others, giving the policy exposure to tighter constraints before it needs to survive them physically.

---

## The Starting Point: What Didn't Work

### Attempt 1: Velocity tracking reward

The first reward was the most natural: reward the robot for moving forward at a target velocity.

```python
reward = exp(-(v_x - v_target)² / 0.25)
```

**The standing-still exploit.** At v_target=0.5 m/s, a stationary robot (v=0) receives `exp(-1) ≈ 0.37` reward — a substantial fraction of the maximum. On narrow bridges where moving is risky, the optimizer found it was safer and often more rewarding to just stand still. This made the reward unreliable below ~0.3m width.

### Attempt 2: Depth observations

We added a 13×13 heightmap observation (depth camera) to give the robot explicit edge awareness. This improved narrow-stage performance but introduced a dangerous dependency: the reward was still weak, so the depth obs was doing the job the reward should have done. When we later improved the reward, the depth obs became unnecessary overhead.

### Attempt 3: Frontier reward with lower scale (v5)

A consultant suggested lowering `frontier_delta` from 50 to 20 and adding stronger penalties. This produced a **bounding gait exploit**: 32 touchdowns/episode vs 198 at scale=50. The optimizer discovered that long hops (few, powerful contacts) earned the same total frontier payout as careful trotting, at less energetic cost. The robot looked like a kangaroo rather than a dog.

**Lesson**: reward scale matters for gait frequency. High scale + contact-gating selects for high-frequency deliberate trotting.

---

## The Foundation: Contact-Gated Frontier Reward (v4/v6)

The breakthrough that made everything downstream possible was replacing velocity tracking with a **contact-gated frontier reward**.

### How it works

Forward progress is *banked* into `unpaid_progress` whenever the robot moves to a new maximum x-position. This banked value is only *paid out* at the moment of foot contact, weighted by foot placement quality:

```python
# Every step: bank forward progress
capped_delta = min(new_max_x - prev_max_x, 0.6 * dt)
unpaid_progress += capped_delta

# At foot contact: pay out all banked progress
frontier_payout = (unpaid_progress / dt) * foothold_multiplier * any_contact
unpaid_progress = 0  # cleared after payout
```

**Why this eliminates the standing-still exploit**: standing still produces zero `unpaid_progress`, so frontier payout is zero. There is no reward for being stationary.

**Why this selects for trotting**: each foot contact triggers a payout. More contacts per unit time = more payouts = more reward. The optimizer naturally discovers high-frequency, deliberate trotting.

**Why scale=50 is critical**: at scale=50, the frontier reward dominates the reward signal and selects for high-frequency contact. At scale=20, a single large hop earns the same total as many small steps, so the optimizer switched to bounding. This was empirically validated — reverting to 50 immediately restored the trot.

### The foothold multiplier

The frontier payout is scaled by a `foothold_multiplier` that rewards precise foot placement:

```python
foothold_multiplier = 0.5 + 0.5 * quality_mean
```

`quality` measures how well-placed each foot is relative to the virtual bridge width. A foot planted near the bridge center earns quality≈1.0; a foot near the edge earns less. This links the forward-progress reward to precise placement.

### v6 results

This reward, combined with a structured curriculum and automated gate logic, produced the first strong result: **0.16m bridge at 92.2% success** without depth observations. Previous best was 0.2m at 89% (with depth). 

| Stage | Width | Success |
|-------|-------|---------|
| 0 | 0.8m | 100% |
| 1 | 0.4m | 100% |
| 2 | 0.32m | 97.7% |
| 3 | 0.24m | 99.2% |
| 4 | 0.16m | 92.2% |
| 5 | 0.10m | **0%** — complete failure |

At 0.10m, the 92% success rate at 0.16m evaporated entirely. The robot couldn't even begin. We initially assumed this was a morphological limit — the bridge narrower than the robot's natural stance. It wasn't. It was a series of reward design flaws.

---

## Diagnosing the 0.16m Ceiling

### Bug 1: Asymmetric world model (heightmap y-grid)

The simulation builds a heightmap to represent the world geometry. The y-axis grid was constructed as:
```python
ys = np.arange(-1.0, 1.0, 0.03)  # BUG: never contains y=0
```

Because `np.arange` with floating-point step can drift, this grid was offset from zero by ~1.5cm. The bridge the policy *perceived* was shifted laterally relative to the bridge that *existed physically*. The left edge appeared 1.5cm further than reality; the right edge appeared 1.5cm closer.

Over hundreds of millions of training steps, the policy anchored its stance to this broken geometry. When evaluated on the true bridge, its feet were systematically off-center. 

**Fix**: Symmetric grid centred exactly on 0:
```python
n_half = int(round((_HM_Y_MAX - _HM_Y_MIN) / 2 / _HM_CELL))
ys = np.arange(-n_half, n_half + 1) * _HM_CELL  # always symmetric
```

**Also fixed**: The `foot_off_bridge` penalty used the same 3cm-resolution heightmap to detect when feet left the bridge. At tight widths this 1cm quantisation error was significant. Replaced with an analytic check:
```python
on_bridge = (foot_x >= 0.0) & (foot_x <= 4.0) & (jp.abs(foot_y) <= hw)
```

### Bug 2: The training-wheels gait (exploit in quality_mean)

The foothold multiplier averaged quality only over *contacting* feet:
```python
foot_on = contact_filt.astype(float)
quality_mean = sum(quality * foot_on) / (sum(foot_on) + 1e-6)
```

**The exploit**: lifting a low-quality foot *removed* it from the average, *raising* the mean quality of the remaining feet. At narrow widths, the policy learned to hover its poorly-placed feet and rely on 2–3 well-placed feet. We observed this in video: the robot used its front-left and back-left legs as primary stance while the right side hovered as stabilisers — essentially bipedal locomotion, ignoring two legs.

Contact monitoring (added as a metric) confirmed: left contact fraction was ~3× right contact fraction at peak. 

**Fix**: average quality over all 4 feet unconditionally:
```python
quality_mean = jp.mean(quality)  # airborne and stance feet included
```
Now lifting a low-quality foot cannot improve the multiplier. The only way to raise it is to physically move all feet inboard.

### Bug 3: Contact-gated penalty created a cadence incentive

The `foot_off_virtual` penalty fired only on contact steps:
```python
return jp.sum(overshoot * contact_filt.astype(float))  # per contact step
```

**The exploit**: bounding (fewer contact events per second) paid the penalty less often than trotting (many contact events). The optimizer had a hidden incentive to reduce contact frequency — to dodge the penalty. This is the opposite of what we want.

**Fix**: make the penalty contact-independent, firing every step on all feet:
```python
return jp.sum(overshoot)  # all feet, every step
```
Scale reduced from −30 to −10 because it now fires continuously (4× more often).

### Bug 4: The dead gradient (the key breakthrough)

This was the most subtle and most important fix. The foothold quality was:
```python
quality = jp.clip(margin / jp.minimum(vhw, 0.10), 0.0, 1.0)
```
where `margin = vhw - |foot_y|` (positive inside virtual_hw, negative outside).

When a foot is *outside* virtual_hw, `margin < 0`, and `quality = clip(..., 0, 1) = 0`. **Zero, not small — exactly zero.** This means:

- Foot at |y|=0.14m with vhw=0.05m → quality = 0
- Foot at |y|=0.08m with vhw=0.05m → quality = 0
- Foot at |y|=0.051m with vhw=0.05m → quality = 0

All three situations look identical to the reward function. Moving from natural stance (0.14m) to the virtual boundary (0.05m) earns zero frontier gradient for 9cm of travel. **The frontier reward had no inboard pull for feet outside the virtual boundary.** The only narrowing signal came from `foot_off_virtual` (penalty) and the fading floor — and they weren't enough.

This is why v6 couldn't get below 0.16m and v8/v9 (which closed the other exploits) also hit a ceiling. The gradient was dead in exactly the regime that mattered most.

**Fix**: soft exponential quality with decay length proportional to vhw:
```python
tau = jp.maximum(0.5 * info["virtual_hw"], 0.01)
quality = jp.exp(jp.minimum(margin, 0.0) / tau)
```

Now quality = 1.0 inside virtual_hw (unchanged), but outside it decays smoothly. At natural stance (|y|=0.14m) against vhw=0.05m:
- Old: quality = 0 (zero gradient)
- New: quality = exp(−0.09/0.025) ≈ 0.03 (small but nonzero, gradient alive)

A small positive quality means the frontier reward has an inboard pull even from natural stance. The policy can feel the gradient and follow it all the way to the virtual target.

---

## The Complete Final Reward Design (v10)

### Core: Contact-gated frontier

```
frontier_payout = (unpaid_progress / dt) × foothold_multiplier × any_contact
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| `unpaid_progress` | Δmax_x banked each step | Converts displacement into credit |
| `foothold_multiplier` | `floor + (1−floor) × quality_mean` | Links reward to precise placement |
| `quality` | `exp(min(margin, 0) / tau)` | Soft gradient pulling feet inboard |
| `floor` | `0.5 × clip((vhw−0.10)/0.05, 0, 1)` | Fades at narrow widths (0 at vhw≤0.10) |
| `any_contact` | binary | Forces foot contact for payout |

The `floor` component deserves explanation: at wide bridges (vhw≥0.15), the floor is 0.5, meaning even a bad stance earns 50% frontier reward. At narrow bridges (vhw≤0.10), floor=0, meaning only genuine inboard foot placement earns any frontier reward at all. This creates increasing pressure to narrow stance as the curriculum progresses.

### Narrowing penalties

| Term | Scale | Behaviour |
|------|-------|-----------|
| `foot_off_virtual` | −10 | Fires every step on all 4 feet; linear hinge on overshoot beyond vhw |
| `feet_stale_air` | −2.0 | Penalises legs held airborne >0.4s (kills training-wheels hover) |

### Standard locomotion penalties

| Term | Scale | Term | Scale |
|------|-------|------|-------|
| `termination` | −200 | `orientation` | −5.0 |
| `foot_off_bridge` | −50 | `lateral_deviation` | −3.0 |
| `feet_clearance` | −2.0 | `heading` | −2.0 |
| `feet_slip` | −0.25 | `lin_vel_z` | −2.0 |
| `dof_pos_limits` | −1.0 | `action_rate` | −0.01 |
| `torques` | −2e-4 | `energy` | −1e-3 |

### Positive shaping

| Term | Scale | Notes |
|------|-------|-------|
| `frontier_delta` | +50 | Contact-gated, quality-weighted |
| `feet_air_time` | +2.0 | Cap at 0.15s — selects trotting, not bounding |
| `success` | +5000 | Large terminal reward for full crossing |
| `pose` | +0.5 | Abduction weight 0.1 (was 1.0 — was competing with narrow-stance signal) |

---

## The Curriculum

Rather than training at one fixed width, we use a structured curriculum: train at each width until a success-rate gate is met, then advance to a narrower bridge. Within each stage, virtual width is sampled per-episode from U[vhw_min, hw], creating diversity.

**Gate logic**: 
- ≥70% success → advance
- 30–70% → extend +100M steps (up to 2 extensions)
- <30% → revert to last good checkpoint, insert midpoint stage

### Final curriculum structure (presentation run)

| Stage | Bridge | hw | vhw_min | Steps | ep_len |
|-------|--------|----|---------|-------|--------|
| 00 | 0.80m | 0.400 | 0.150 | 150M | 1000 |
| 01 | 0.40m | 0.200 | 0.100 | 100M | 1000 |
| 02 | 0.32m | 0.160 | 0.080 | 100M | 1000 |
| 03 | 0.24m | 0.120 | 0.060 | 150M | 1000 |
| 04 | 0.20m | 0.100 | 0.050 | 150M | 1500 |
| 05 | 0.16m | 0.080 | 0.050 | 200M | 1500 |
| 06 | 0.13m | 0.065 | 0.045 | 150M | 1500 |
| 07 | 0.10m | 0.050 | 0.040 | 200M | 1500 |
| 08 | 0.09m | 0.045 | 0.035 | 200M | 1500 |
| 09 | 0.08m | 0.040 | 0.030 | 200M | 1500 |
| 10 | 0.07m | 0.035 | 0.028 | 200M | 1500 |
| 11 | 0.06m | 0.030 | 0.028 | 200M | 2000 |
| 12 | 0.055m | 0.0275 | 0.0275 | 200M | 2000 |

**Note on vhw_min**: at the narrowest stages vhw_min is set *below* hw (or equal) so the virtual reward target stays tighter than the physical boundary. This ensures `foot_off_virtual` and the frontier gradient remain active even when feet reach the physical edge. At stages 10–12, vhw_min is floored at foot_radius + 0.005 = 0.028m to avoid sampling virtual widths where even a centred foot incurs an unsatisfiable penalty.

---

## Results

### v10 (proof-of-concept run, completed 2026-06-13)

| Stage | Bridge | Success | Touchdowns/ep | Contact L/R |
|-------|--------|---------|--------------|-------------|
| 0 | 0.8m | 100% | 91 | — |
| 1 | 0.4m | 100% | 83 | — |
| 2 | 0.32m | 100% | 157 | — |
| 3 | 0.24m | 99.2% | 114 | — |
| 4 | 0.20m | 100% | 130 | 312 / 230 |
| 5 | 0.16m | 100% | 175 | 283 / 223 |
| 6 | 0.13m | 98.4% | 131 | 313 / 224 |
| 7 | 0.10m | **91.4%** | 171 | 274 / 202 |

**Previous best**: v6 at 0.16m (92%). v10 more than doubles the difficulty, reaching 0.10m at 91%.

### Sub-0.10m exploration

After establishing 0.10m as passable, we continued the curriculum to find the morphological limit:

| Stage | Bridge | Success |
|-------|--------|---------|
| 8 | 0.09m | 96.9% |
| 9 | 0.08m | 99.2% |
| 10 | 0.07m | 98.4% |
| 11 | 0.06m | **97.7%** |
| 12 | 0.055m | **0%** — failure |

The robot passed 0.06m at 97.7% — a bridge only **6cm wide**, with just 7mm clearance between the foot sphere and the edge. At 0.055m (4.5mm clearance), it failed completely.

**The cliff is sharp**: 97.7% → 0% in a single 5mm step. This is not a curriculum gap — the gate logic would have inserted a midpoint if performance was marginal. The 0.055m bridge appears to be near the practical morphological limit for the Go1 in simulation, driven by roll stability (the support polygon becomes too narrow to reject normal gait perturbations) and foot-fit geometry.

For context: at 0.06m bridge width, the robot's support polygon has a lateral extent of only 6cm at a CoM height of ~30cm — a lateral stability margin of roughly 6° before tipping. At 0.055m this shrinks to ~5°, below the perturbation tolerance.

---

## Key Lessons for Reward Design

1. **Exploit hunting is the core work.** Every time performance plateaued, it was because the optimizer had found a shortcut we hadn't anticipated. The reward design story is largely a story of closing exploits.

2. **Contact-gating is powerful but exploitable.** Any reward that fires only at contact can be gamed by changing contact frequency. The training-wheels exploit (fewer feet, more quality) and the cadence exploit (bounding to dodge contact-step penalties) both came from this. The fix in both cases: move signals to be contact-independent.

3. **Dead gradients are invisible.** The quality clip-at-0 bug produced no obvious failure mode — the policy still learned *something* at every stage. It just hit a ceiling earlier than necessary. Without explicitly tracing the gradient through the reward, we'd never have found it.

4. **World model correctness matters more at tight constraints.** A 1.5cm offset is negligible on a 0.8m bridge (1.9%) but significant on a 0.10m bridge (15%). Bugs that are harmless at one scale become load-bearing at another.

5. **Virtual curriculum (vhw sampling) is essential.** Without per-episode width randomization, the policy memorizes a specific stance for the physical width. With it, the policy learns to track the virtual width signal and generalize continuously — as evidenced by the `max_foot_y` metric decreasing monotonically across stages.

---

## What's Next

- **Presentation run** (`bridge_crossing_final`, currently training): clean end-to-end run from 0.8m to 0.055m for full data in one WandB project.
- **Physical hardware transfer**: simulation target was 0.13m — we hit 98.4% there. Sim-to-real gap testing is the next step.
- **Inboard-pull quality** (see `inboard_pull_quality_idea.md`): a potential future improvement that would give the frontier reward a live gradient *inside* the virtual band, not just outside it. May push the morphological limit further.
- **Humanoid robots**: the same environment and reward structure generalises to bipedal robots (H1, G1). Narrow bridge crossing for bipeds is an open problem.
