# Reward Design for Narrow Bridge Crossing: A Technical Account

**Project**: Narrow Bridge Locomotion — Unitree Go1 Quadruped  
**Simulation**: MuJoCo MJX · Brax PPO · 4096 parallel environments  
**Date**: 2026-06-16  
**Result**: 0.06m bridge (6cm total width) crossed at 97.7% success rate in simulation; morphological limit identified at 0.055m

---

## Abstract

We present the reward engineering process that enabled a Unitree Go1 quadruped to cross bridges as narrow as 6cm in simulation — a width less than one quarter of the robot's natural stance. The robot uses only proprioceptive observations (no depth camera, no external positioning). We document five reward exploit failures, two world model bugs, and a dead-gradient diagnosis, and show how each fix contributed to the final result. The core insight is that a contact-gated frontier reward with a soft exponential quality function provides a continuous inboard gradient that velocity-tracking and hard-clipped quality formulations cannot supply.

---

## 1. Task Definition

### 1.1 Environment

The scene contains three regions:

- **Platform A** (starting): 3m × 2m, centred at x ∈ [−3, 0]
- **Bridge**: 4m long, half-width `hw` configurable, x ∈ [0, 4]
- **Platform B** (goal): 3m × 2m, centred at x ∈ [4, 7]

The robot spawns at a randomised pose on Platform A facing the bridge. The task is to cross the bridge and reach Platform B. Falling off any surface terminates the episode. A reach-goal bonus (+5000) fires once the robot's base crosses a threshold on Platform B.

### 1.2 Morphological constraints

| Quantity | Value |
|----------|-------|
| Natural stance half-width (foot y from centreline) | ~0.14 m |
| Hip-to-hip spacing | ~0.18 m |
| Foot collision sphere radius (`_FOOT_RADIUS`) | 0.023 m |
| Min. foot-centre clearance at hw=0.05m | 0.027 m |

The robot's natural stance (0.28m total) is wider than every bridge we train on below Stage 1. Learning to cross requires genuine kinematic adaptation, not just careful stepping.

### 1.3 MDP formulation

**State / Observation** (46-dimensional, proprioceptive only):

| Group | Dims | Signal |
|-------|------|--------|
| Joint positions | 12 | Relative to default pose |
| Joint velocities | 12 | rad/s |
| Base linear velocity (local) | 3 | m/s |
| Base angular velocity (local) | 3 | rad/s |
| Gravity vector (local) | 3 | Orientation proxy |
| Foot contact booleans | 4 | Binary |
| Goal direction (local) | 2 | Unit vector toward goal |
| Virtual bridge half-width | 1 | Normalised by 0.4 |
| Previous action | 12 | t−1 joint targets |

All observations are available on real hardware (encoders, IMU, foot force sensors). No depth camera, no heightmap, no external localisation.

**Action space** (12-dimensional): joint position targets, applied via PD control (Kp=20, Kd=0.5 for most joints). Control frequency 50 Hz (dt = 0.02 s).

**Episode termination**:
- Base contacts non-foot geometry (fall)
- Base orientation > 0.7 rad from upright
- Foot leaves any surface (foot_off_bridge penalty triggers; fall usually follows)
- Episode timeout (ep_len steps, 1000–2000 depending on stage)
- Success: base x > 5.5 m

**Training**: Brax PPO, γ = 0.997, entropy cost = 0.01, 4096 parallel environments.

---

## 2. Virtual Bridge Width Curriculum

A key structural element, introduced early and retained throughout, is the **virtual bridge half-width** (`virtual_hw` or `vhw`). At the start of each episode, a virtual width is sampled:

$$\text{vhw} \sim \mathcal{U}[\text{vhw\_min},\ hw]$$

where `hw` is the fixed physical half-width for that curriculum stage. The reward and quality signals use `vhw`; the physics (fall termination) uses `hw`. This means:

- The robot trains against a range of difficulties within each stage.
- In episodes where `vhw < hw`, the robot is rewarded for a narrower stance than strictly necessary — pre-conditioning it for the next stage.
- The policy observes `vhw` (normalised), so it can adapt its behaviour to the sampled episode difficulty.

Critically, `vhw_min` is set **below** `hw` at the narrowest curriculum stages (not just equal to it). This ensures the reward gradient remains active when feet are at the physical boundary — setting `vhw_min = hw` would make every episode sample `vhw = hw`, eliminating the within-stage curriculum.

---

## 3. Reward Design: Evolution and Exploits

The reward went through five generations, each motivated by an exploit or failure mode found in the previous version. We document each in sequence.

### 3.1 Generation 0: Velocity tracking (failed)

**Formulation:**
$$r_{\text{vel}} = \exp\!\left(-\frac{(v_x - v^*)^2}{0.25}\right)$$

with target velocity $v^* = 0.5$ m/s.

**Exploit: standing-still.** At $v_x = 0$, $r_{\text{vel}} = e^{-1} \approx 0.37$ — a substantial reward for doing nothing. On narrow bridges where forward motion is dangerous, the policy discovered that standing still was frequently the risk-minimising strategy. Below ~0.3m bridge width, training consistently produced stationary robots. Abandoned.

### 3.2 Generation 1: Contact-gated frontier reward

This is the foundational design that all subsequent versions build on.

**Motivation**: We want to reward forward progress, but in a way that (a) gives zero reward for being stationary, and (b) incentivises foot contact. Velocity tracking fails both criteria. A frontier-based reward — reward new x-maximum reached — satisfies (a) but not (b). A contact-gate on top satisfies both.

**Implementation:**

At each step, forward progress is banked:
$$\text{unpaid} \mathrel{+}= \min\!\left(\max_x - \text{prev\_max}_x,\ 0.6 \cdot \Delta t\right)$$

The cap ($0.6 \cdot \Delta t = 0.012$ m/step) prevents windfalls from large jumps.

At foot-contact steps, the banked progress is paid out and cleared:
$$r_{\text{frontier}} = \frac{\text{unpaid}}{\Delta t} \cdot \mu \cdot \mathbf{1}[\text{any contact}]$$
$$\text{unpaid} \leftarrow 0$$

where $\mu$ is the `foothold_multiplier` (quality gate, detailed in §3.3).

**Why scale=50 matters.** The scale was empirically critical:

| Scale | Behaviour | Touchdown/ep | Abduction sat |
|-------|-----------|-------------|---------------|
| 20 | Bounding (long hops) | 32 | 24% |
| 50 | Deliberate trot | 198 | 8.8% |

At scale 20, a single large hop yields the same total frontier payout as many small steps (same total Δx). The optimizer switches to low-frequency bounding. At scale 50, the frontier term dominates enough that high-frequency contact is selectively favoured — each contact event triggers a payout, so more contacts per unit time equals more total reward.

**Air time cap at 0.15s.** The `feet_air_time` reward:
$$r_{\text{air}} = \sum_i \left(\min(\tau_i^{\text{air}},\ 0.15) - 0.10\right) \cdot \mathbf{1}[\text{first contact}_i]$$

caps at 0.15s, not 0.25s (an early regression used 0.25s). Above 0.15s, longer flights are directly rewarded — this selects for bounding. The 0.15s cap saturates quickly and rewards short-flight trotting.

### 3.3 The foothold multiplier: four generations

The multiplier $\mu$ gates the frontier payout on foot placement quality. Its evolution is the central reward engineering story.

#### Gen 1 (v6): Contact-only quality average

$$\text{quality}_i = \text{clip}\!\left(\frac{\text{vhw} - |y_i|}{0.10},\ 0,\ 1\right)$$
$$\mu = 0.5 + 0.5 \cdot \frac{\sum_i q_i \cdot c_i}{\sum_i c_i + \epsilon}$$

where $c_i \in \{0,1\}$ is foot contact, $y_i$ is foot lateral position.

**Exploit: training-wheels gait.** Averaging quality over *contacting* feet only created a perverse incentive: lifting a low-quality foot **removed** it from the denominator, *raising* the average. At narrow bridges where some feet were far outside `vhw`, the policy discovered it could improve $\mu$ by hovering poorly-placed feet and maintaining stance with 2–3 well-placed feet. We observed this as a persistent left-biased gait: the robot used FL+RL as primary stance while FR+RR hovered as stabilisers (contact ratio 3:1 left:right). This is essentially 2-legged locomotion.

#### Gen 2 (v8/v9): All-feet quality average + fading floor

$$\mu = \text{floor} + (1 - \text{floor}) \cdot \frac{1}{4}\sum_{i=1}^4 q_i$$
$$\text{floor} = 0.5 \cdot \text{clip}\!\left(\frac{\text{vhw} - 0.10}{0.05},\ 0,\ 1\right)$$

The all-feet average closes the lifting exploit: airborne feet count toward the mean, so lifting a low-quality foot cannot raise $\mu$.

The `floor` provides a non-zero baseline at wide bridges (floor=0.5 when vhw≥0.15m) that fades to zero at narrow bridges (floor=0 when vhw≤0.10m). At narrow stages, only genuine inboard placement earns frontier reward — there is no free 50%.

**Exploit: cadence exploit (from `foot_off_virtual`).**

At this stage we added a lateral penalty:
$$r_{\text{off\_virt}} = -30 \cdot \sum_i \max(|y_i| + r_{\text{foot}} - \text{vhw},\ 0) \cdot c_i$$

This fired *per contact step*. Bounding (fewer contact events) therefore paid this penalty less often than trotting. The optimizer had a hidden incentive to reduce gait frequency — the opposite of what the frontier reward selects for.

#### Gen 3 (v9): Contact-independent penalty

$$r_{\text{off\_virt}} = -10 \cdot \sum_{i=1}^4 \max(|y_i| + r_{\text{foot}} - \text{vhw},\ 0)$$

Now fires every step on all four feet, regardless of contact. Swing and stance feet pay equally. No cadence incentive. Scale reduced from −30 to −10 because it now fires ~4× more often (4 feet, every step vs 2 feet, contact steps only).

#### Gen 4 (v10): Soft exponential quality — the key fix

**The dead-gradient problem.** The clip in the quality function:
$$q_i = \text{clip}\!\left(\frac{\text{vhw} - |y_i|}{0.10},\ 0,\ 1\right)$$

evaluates to exactly **0** whenever $|y_i| > \text{vhw}$. This means:

$$\frac{\partial \mu}{\partial y_i} = 0 \quad \text{whenever } |y_i| > \text{vhw}$$

The frontier reward has **zero gradient** with respect to inboard foot movement for any foot outside the virtual boundary. At natural stance ($|y_i| \approx 0.14$ m) against a narrow virtual width (vhw = 0.05 m), all feet are outside, so $q_i = 0$ for all $i$, $\mu = \text{floor}$, and the frontier term carries no narrowing signal whatsoever.

This explains why v6 hit a ceiling at 0.16m and why v8/v9 (which closed the other exploits) still hit a ceiling: the primary reward term provided no inboard pull for feet in exactly the regime that needed it most — far outside the virtual boundary at narrow stages.

**Fix: soft exponential quality**

$$\tau = \max(0.5 \cdot \text{vhw},\ 0.01)$$
$$q_i = \exp\!\left(\frac{\min(\text{vhw} - |y_i|,\ 0)}{\tau}\right)$$

**Properties:**
- $q_i = 1.0$ for $|y_i| \leq \text{vhw}$ (unchanged — feet inside the band get full quality)
- $q_i = e^{-1} \approx 0.37$ at the virtual boundary ($|y_i| = \text{vhw}$)
- Decays smoothly for $|y_i| > \text{vhw}$ with length scale $\tau = 0.5 \cdot \text{vhw}$
- $\frac{\partial q_i}{\partial |y_i|} = -q_i / \tau \neq 0$ for all $y_i$ — **gradient always live**

At natural stance vs vhw=0.05m:
$$q_i = \exp\!\left(\frac{0.05 - 0.14}{0.025}\right) = \exp(-3.6) \approx 0.027$$

Small, but nonzero. The frontier reward now provides a weak but continuous pull from any stance width. Combined with $r_{\text{off\_virt}}$ (a stronger signal closer to the boundary), the policy has gradient information from the very first training step.

The scaling $\tau = 0.5 \cdot \text{vhw}$ ensures the decay length adapts proportionally at narrow stages: at vhw=0.025m (stage 11), $\tau = 0.0125$m, giving sharper discrimination at smaller scales than a fixed 5cm length constant would provide.

### 3.4 Additional penalties

**`feet_stale_air`** (added v9):
$$r_{\text{stale}} = -2.0 \cdot \sum_{i=1}^4 \max(\tau_i^{\text{air}} - 0.4,\ 0)$$

Penalises legs held airborne for more than 0.4 seconds. A normal trot swing phase is ~0.2s, comfortably below the threshold. This penalty specifically targets the training-wheels pattern (persistently hovering a leg), without penalising normal gait dynamics. The threshold was chosen to give ~0.2s headroom above normal swing duration.

**`pose` abduction weight** (v9): the pose regularisation term:
$$r_{\text{pose}} = \exp\!\left(-\sum_j w_j (q_j - q_j^*)^2\right)$$

originally weighted abduction joints equally with hip and knee ($w_{\text{ab}} = 1.0$). At narrow stages, the narrow-stance signal requires sustained abduction — the robot must deviate from the default (wide) pose. The abduction weight was reduced from 1.0 to 0.1, removing what was effectively an adduction tax that competed with the quality gradient.

### 3.5 World model corrections

**Bug: asymmetric heightmap grid.** The environment maintained an internal heightmap for reward computation. The y-axis grid was:
```python
ys = np.arange(-1.0, 1.0, 0.03)  # floating point drift: never contains y=0
```
Due to floating-point accumulation in `np.arange`, this grid was offset from zero by approximately 1.5cm. The policy's perceived bridge was asymmetric: the left edge appeared further than reality, the right edge closer. Over hundreds of millions of training steps, the policy anchored its lateral stance to this erroneous geometry.

This manifested as persistent left-right contact asymmetry (contact_left ≈ 2.8× contact_right in v8) and likely contributed to the 0.16m ceiling — the policy's learned stance was calibrated to a bridge that didn't exist at the correct position.

**Fix:** Symmetric grid centred exactly on 0:
```python
n_half = int(round((_HM_Y_MAX - _HM_Y_MIN) / 2 / _HM_CELL))
ys = np.arange(-n_half, n_half + 1) * _HM_CELL
```

**Bug: heightmap-based foot_off_bridge.** The penalty for stepping off the bridge used heightmap lookup:
$$r_{\text{off\_bridge}} = -50 \cdot \sum_i \mathbf{1}[\text{foot}_i \notin \text{surface}]$$

where surface membership was determined by indexing into the 3cm-resolution heightmap. At tight bridge widths, the 3cm cell introduces up to ~1cm positional error — roughly 20% of the 0.06m bridge half-width at the narrowest stages.

**Fix:** Analytic surface check with exact boundaries:
```python
on_bridge = (foot_x >= 0.0) & (foot_x <= 4.0) & (jp.abs(foot_y) <= hw)
```
No quantisation error.

---

## 4. Complete Reward Function (Final)

The full per-step reward is:

$$r = \Delta t \cdot \text{clip}(R,\ -10,\ 10000)$$

where $R$ is the unscaled sum:

$$R = \sum_k s_k \cdot r_k$$

### Positive terms

| Term | Scale | Formulation |
|------|-------|-------------|
| `frontier_delta` | +50 | $({\text{unpaid}}/{\Delta t}) \cdot \mu \cdot \mathbf{1}[\text{any contact}]$ |
| `feet_air_time` | +2.0 | $\sum_i \max(\min(\tau_i^{\text{air}}, 0.15) - 0.10, 0) \cdot \mathbf{1}[\text{first contact}_i]$ |
| `success` | +5000 | $\mathbf{1}[x_{\text{base}} > 5.5]$ (terminal) |
| `pose` | +0.5 | $\exp(-\sum_j w_j (q_j - q_j^*)^2)$, $w_{\text{ab}}=0.1$, $w_{\text{hip,knee}}=1.0$ |

### Negative terms (penalties)

| Term | Scale | Formulation |
|------|-------|-------------|
| `termination` | −200 | $\mathbf{1}[\text{fall or contact violation}]$ (terminal) |
| `foot_off_bridge` | −50 | $\sum_i \mathbf{1}[\text{first contact}_i \wedge \text{foot}_i \notin \text{surface}]$ |
| `foot_off_virtual` | −10 | $\sum_i \max(|y_i| + r_{\text{foot}} - \text{vhw}, 0)$ (all feet, every step) |
| `feet_stale_air` | −2.0 | $\sum_i \max(\tau_i^{\text{air}} - 0.4, 0)$ |
| `orientation` | −5.0 | $\|g_{\text{local}} - g_0\|^2$ |
| `lateral_deviation` | −3.0 | $y_{\text{base}}^2$ |
| `heading` | −2.0 | Deviation from bridge axis |
| `lin_vel_z` | −2.0 | $v_z^2$ |
| `feet_clearance` | −2.0 | Foot height error during swing |
| `feet_slip` | −0.25 | $\sum_i \|v_i^{\text{foot}}\|^2 \cdot c_i$ |
| `dof_pos_limits` | −1.0 | Joint limit violation |
| `action_rate` | −0.01 | $\|a_t - a_{t-1}\|^2$ |
| `torques` | −2×10⁻⁴ | $\|\tau\|^2$ |
| `energy` | −1×10⁻³ | $\|\tau \cdot \dot{q}\|$ |
| `feet_height` | −0.2 | Foot height error at stance |

### Foothold multiplier (complete)

$$\tau = \max(0.5 \cdot \text{vhw},\ 0.01)$$
$$q_i = \exp\!\left(\frac{\min(\text{vhw} - |y_i|,\ 0)}{\tau}\right), \quad i = 1,\ldots,4$$
$$\text{quality\_mean} = \frac{1}{4}\sum_{i=1}^4 q_i$$
$$\text{floor} = 0.5 \cdot \text{clip}\!\left(\frac{\text{vhw} - 0.10}{0.05},\ 0,\ 1\right)$$
$$\mu = \text{floor} + (1 - \text{floor}) \cdot \text{quality\_mean}$$

---

## 5. Curriculum Structure

### 5.1 Gate logic

Each stage runs for a budget of training steps, then evaluates `metric/term_success` (fraction of episodes that reached Platform B):

- **≥ 0.70**: advance to next stage
- **[0.30, 0.70)**: extend +100M steps (up to 2 extensions), re-evaluate
- **< 0.30**: revert to last-passing checkpoint, insert midpoint stage between last good and current, exit for manual re-launch

The gate threshold of 0.70 was chosen to allow advancement while not requiring convergence — empirically, policies that pass at 0.70 continue improving when transferred to the next stage.

### 5.2 Stage schedule (final presentation curriculum)

| Stage | Bridge | hw (m) | vhw\_min (m) | Steps | ep\_len |
|-------|--------|--------|-------------|-------|---------|
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

**Note on vhw\_min flooring.** Below hw ≈ 0.038m, the `foot_off_virtual` penalty becomes structurally non-zero for a perfectly centred foot if `vhw < foot_radius`:
$$\text{overshoot} = |y_i| + 0.023 - \text{vhw} > 0 \quad \text{even when } y_i = 0$$
This corrupts the advantage signal by creating an unconditional per-step penalty the policy cannot zero out. To prevent this, `vhw_min` is floored at $r_{\text{foot}} + 0.005 = 0.028$m for stages where $hw - 0.010 < 0.028$m, effectively decoupling `vhw_min` from `hw` at the narrowest stages.

### 5.3 Known midpoint insertions

Two curriculum failures required midpoint insertions and re-launch:

**v6 → 0.10m**: the jump from 0.16m (hw=0.08) to 0.10m (hw=0.05) failed at 0% — too steep. A 0.13m midpoint (hw=0.065) was inserted (Stage 05/06 in the final schedule).

**v10 first attempt → 0.16m**: the jump from 0.24m (hw=0.12) to 0.16m (hw=0.08) produced 75.8% success that collapsed to 0% by the end of training. A 0.20m midpoint (hw=0.10) was inserted (Stage 04 in the final schedule). The collapse was attributed to re-emergence of the right-biased training-wheels gait late in the training run, combined with the 33% width reduction being too aggressive for the new reward dynamics.

---

## 6. Results

### 6.1 Success rate across widths (v10 proof-of-concept run)

| Stage | Bridge | Success | TDs/ep | Contact L/R |
|-------|--------|---------|--------|-------------|
| 0 | 0.80m | 100% | 91 | — |
| 1 | 0.40m | 100% | 83 | — |
| 2 | 0.32m | 100% | 157 | — |
| 3 | 0.24m | 99.2% | 114 | — |
| 4 | 0.20m | 100% | 130 | 312/230 |
| 5 | 0.16m | 100% | 175 | 283/223 |
| 6 | 0.13m | 98.4% | 131 | 313/224 |
| 7 | 0.10m | **91.4%** | 171 | 274/202 |

Previous best: 0.16m at 92.2% (v6, with dead-gradient quality). The soft quality fix unlocked everything below 0.16m.

### 6.2 Sub-0.10m morphological probe

| Stage | Bridge | hw | Slack* | Success |
|-------|--------|----|--------|---------|
| 8 | 0.09m | 0.045 | 22mm | 96.9% |
| 9 | 0.08m | 0.040 | 17mm | 99.2% |
| 10 | 0.07m | 0.035 | 12mm | 98.4% |
| 11 | 0.06m | 0.030 | 7mm | **97.7%** |
| 12 | 0.055m | 0.0275 | 4.5mm | **0%** |

*Slack = hw − foot\_radius = foot-centre clearance to bridge edge.

The cliff between 0.06m and 0.055m is sharp: 97.7% collapses to 0% in a single 5mm step. This is not a curriculum gap — the gate logic would have inserted a midpoint for marginal performance (30–70%). A true failure (<30%) indicates the task became suddenly unsolvable.

### 6.3 Morphological limit analysis

**The Go1's geometry at 0.06m bridge (hw=0.030m):**
- Required foot-centre lateral bound: |y| < 0.030m
- Foot sphere at edge: y + 0.023 = 0.053m from centre
- Support polygon half-width at 4-foot contact: ~0.030m
- CoM height: ~0.30m
- Lateral stability margin: arctan(0.030/0.30) ≈ **5.7°**

At 0.055m (hw=0.0275m), the lateral stability margin drops to ~5.3°. The difference is small in degrees but the success rate collapses entirely, suggesting the Go1's gait perturbations regularly exceed ~5° roll at these bridge widths — which are below the support polygon stability threshold.

Secondary contributing factor: at vhw < foot_radius, the penalty formulation becomes degenerate (§5.2), which limits how much lower we can probe without reward redesign.

### 6.4 Contact symmetry as gait quality indicator

The contact symmetry metric (fraction of steps each side is in contact) tracked the disappearance and prevention of lateral gait asymmetry:

| Version | Condition | Contact L/R ratio |
|---------|-----------|-------------------|
| v8 (training-wheels exploit) | Peak performance | 2.8 : 1 |
| v9 (all-feet quality + stale_air) | Any stage | 1.3–1.4 : 1 |
| v10 (all fixes) | All stages | 1.3–1.4 : 1 |

A ratio of 1.3–1.4 persists across all v10 stages even after the world model was made symmetric. This is likely a gait preference (diagonal left-leading trot) rather than a reward artefact — worth monitoring in hardware transfer as a potential source of lateral drift.

### 6.5 Stance narrowing: max\_foot\_y progression

`metric/max_foot_y` (summed over episode steps, proportional to mean stance half-width × episode length):

| Stage | Bridge | max\_foot\_y (sum) |
|-------|--------|--------------------|
| 0 | 0.80m | 76.1 |
| 3 | 0.24m | 29.2 |
| 5 | 0.16m | 28.3 |
| 6 | 0.13m | 21.6 |
| 7 | 0.10m | 23.3 |

The monotonic decrease confirms the policy is genuinely narrowing its stance across the curriculum rather than memorising a fixed gait. At 0.10m, the mean foot half-width is approximately 23.3 / (200 × 0.02 × freq) — the policy is actively tracking `virtual_hw`.

---

## 7. Exploit Taxonomy

Five distinct exploits were found and closed during development. We summarise them in causal order:

| # | Name | Mechanism | Observable symptom | Fix |
|---|------|-----------|-------------------|-----|
| 1 | Standing-still | Velocity reward nonzero at v=0 | Robot stationary on narrow bridges | Replace with contact-gated frontier |
| 2 | Bounding | Low frontier scale made hops equal to taps in total reward | 32 TD/ep, kangaroo locomotion | frontier\_delta scale=50 |
| 3 | Training-wheels | Contact-only quality\_mean; lifting bad foot raised multiplier | 3:1 left:right contact ratio | All-feet unconditional quality\_mean |
| 4 | Cadence exploit | Contact-gated penalty; bounding paid less penalty | Bounding re-emerged alongside exploit 3 | Contact-independent penalty, all feet every step |
| 5 | Dead gradient | Hard clip set quality=0 outside virtual\_hw; zero inboard gradient | Ceiling at 0.16m despite fixes 3+4 | Soft exponential quality |

Exploits 3, 4, and 5 were not independent — exploit 3 triggered the introduction of the virtual width penalty that created exploit 4, and the fix for exploit 4 revealed exploit 5 as the binding constraint. The ordering mattered: fixing them out of sequence would have masked the downstream issues.

---

## 8. Privileged State and Sim-to-Real Considerations

### 8.1 Observation vs. reward state

A critical distinction for hardware transfer:

| Signal | Used in | Available on hardware? |
|--------|---------|----------------------|
| Joint pos/vel | Observation + reward | ✅ Encoders |
| IMU (lin/ang vel, orientation) | Observation | ✅ Onboard IMU |
| Foot contact booleans | Observation + reward | ✅ Force sensors |
| Goal direction | Observation | ✅ From rough localisation |
| Virtual bridge half-width | Observation | ✅ Set manually |
| Foot y-positions (`y_i`) | Reward only | ❌ Requires motion capture or FK |
| Robot x-position (`max_x_reached`) | Reward only | ❌ Requires localisation |
| Lateral deviation | Reward only | ❌ Requires localisation |

The **policy observation space** is entirely hardware-accessible. The **reward** uses privileged simulation state, but rewards are only needed during training — not at deployment.

**Consequence**: the trained policy can be deployed zero-shot on real hardware (observations → network → joint commands). However, it cannot be fine-tuned or trained from scratch on hardware without external sensing (motion capture or LiDAR localisation for foot and body position).

### 8.2 Sim-to-real gap concerns

The reward shaping may implicitly create behaviors that are sim-specific:

- **Foot placement precision**: the quality gradient rewards sub-centimetre lateral precision. Real hardware has motor position noise, gear compliance, and ground contact uncertainty. The policy may have learned to rely on precision that doesn't exist in reality.
- **Contact detection**: foot contact booleans in simulation are noiseless. Real force sensors threshold continuous contact forces — fast steps may trigger brief contact events that don't match the simulation's binary model.
- **No domain randomisation**: these runs used no DR (no friction randomisation, no mass perturbation, no motor latency). The gap to real hardware is unknown.

### 8.3 Path to hardware training

If fine-tuning on hardware is desired, the reward must be reconstructed from observable signals:

| Current term | Hardware-accessible approximation |
|-------------|----------------------------------|
| `frontier_delta` (uses x-position) | Visual odometry or wheel odometry |
| `foot_off_virtual` (uses foot y) | Forward kinematics from joint angles |
| `quality` / `foothold_multiplier` | Forward kinematics (approximate) |
| `lateral_deviation` | From odometry or IMU integration |
| `foot_off_bridge` | Falls are self-evidencing (termination signal) |

Forward kinematics from joint angles provides approximate foot positions without external sensing — the accuracy is limited by leg compliance and ground contact uncertainty but may be sufficient.

---

## 9. Summary of Contributions

1. **Contact-gated frontier reward** with scale=50 eliminates the standing-still exploit and selects for high-frequency deliberate trotting.

2. **Soft exponential quality function** with vhw-proportional length scale provides continuous inboard gradient from any stance width — the critical fix that unlocked sub-0.16m performance.

3. **All-feet unconditional quality average** closes the training-wheels exploit without changing the quality signal for well-placed feet.

4. **Contact-independent `foot_off_virtual` penalty** eliminates the cadence exploit introduced when penalising per-contact-step.

5. **`feet_stale_air` penalty** with 0.4s threshold specifically targets persistent leg hovering without affecting normal gait dynamics.

6. **Symmetric heightmap grid** corrects a 1.5cm lateral asymmetry in the world model that biased learned stance geometry.

7. **Analytic `foot_off_bridge`** replaces quantised heightmap lookup with exact geometric check.

8. **Virtual curriculum with `vhw_min < hw`** trains the policy against a range of difficulties within each stage and pre-conditions for the next stage, with the floor preventing degenerate virtual widths at the narrowest stages.

These contributions together enabled crossing bridges from 0.80m (prior art: 0.20m in this codebase) down to 0.06m, representing a 10× reduction in bridge width relative to the robot's natural stance.

---

## Appendix: Exploit Detection Methodology

Each exploit was detected through a combination of:

1. **Quantitative metrics**: `touchdown_count`, `metric/contact_left`, `metric/contact_right`, `metric/abduction_saturation` logged to WandB every eval interval.
2. **Video inspection**: the MuJoCo viewer was used to observe qualitative gait patterns when metrics were anomalous.
3. **Reward gradient analysis**: tracing the mathematical derivative of the reward with respect to the decision variable (foot position, contact pattern, gait frequency) to identify zero-gradient regions before training.

Method (3) was the most valuable and underused early in the project. The dead-gradient problem (Exploit 5) was only found by explicitly computing $\partial \mu / \partial y_i$ and observing that it was zero in the relevant regime — a purely analytical step that required no training runs.
