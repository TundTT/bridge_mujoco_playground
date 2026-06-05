# Plan: Local Heightmap Observation for Bridge Edge Perception

> **Status: COMPLETED — 2026-06-04**
>
> Addresses the perception wall hit at 0.3m bridge width. Implementation in `bridge.py`
> (commit a455765). All 3 test stages passed. Experiment 2 results: 91% at 0.3m (+42 pp),
> 89% at 0.2m (+73 pp). 0.1m confirmed as a morphological wall, not a perception problem.

---

## The Problem

The current `state` observation has no y-position and no explicit bridge-edge signal.
The policy learns a gait *prior* that stays centred on average — which degrades to ~50/50
at narrow widths. At 0.1m the robot tightrope-walks competently but never crosses because
it cannot detect drift and correct early enough.

Giving the robot a local view of the terrain geometry directly solves this: it can see
"bridge surface is here, void is here", detect which direction it's drifting, and correct.

---

## The Idea

A **local heightmap observation** — the standard approach for terrain-aware locomotion RL:

1. Represent the bridge scene as a 2D grid of terrain heights (bridge surface height vs. void/fall-off height).
2. At every step in `_get_obs()`, sample a small NxN patch of that grid centred on the robot's current (x, y) position.
3. Add the flattened patch (169 floats) to the policy's `state` observation vector.

The robot's position within the patch immediately encodes "edge is N cells to the left/right",
giving it the reactive centering reflex that's currently missing.

This was suggested as a "world depth map + robot position in it" — the heightmap is the
depth map of the world, and indexing into it by robot position is how the robot knows where
it sits within that map.

---

## Two Heightmaps — Terminology

**World heightmap** (`self._world_heightmap`) — the full pre-computed 2D grid of the entire
scene, stored in fixed world (x, y) coordinates. Built once at `__init__`. Never changes
during an episode. Never put directly into the observation. This is the lookup table.

**Local heightmap** (`local_heightmap` in `_get_obs()`) — the 13×13 patch sampled from the
world heightmap each step, rotated so that "row 0 = ahead of the robot, column 0 = robot's
left" regardless of which way the robot faces in the world. This is what the policy sees.
Recomputed every timestep.

The key distinction: the world heightmap is indexed in world (x,y) — but *which cells get
sampled* is determined by the robot's yaw. When the robot turns, the local sample grid
rotates with it, so the local heightmap always shows terrain from the robot's point of view.
The world heightmap itself never rotates; only the window into it does.

---

## Why This Works in MJX

- The world heightmap is static within an episode — no per-step rendering needed.
- Sampling it is just array indexing — fully JIT-compilable and vectorised across the batch.
- The world heightmap only needs rebuilding when `bridge_half_width` changes (once per curriculum stage).

No camera, no image, no GPU rendering pipeline.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Patch size | **13×13** | Must cover feet + lookahead past them; see resolution analysis. |
| Grid resolution | **3cm/cell** | Balances foot resolution vs. patch coverage; see analysis. |
| Coordinate frame | **Robot-local (rotated by yaw)** | Policy sees same spatial pattern regardless of heading. |
| Height encoding | **Binary (1.0 = surface, 0.0 = void)** | Simplest; all terrain surfaces are at the same height (z=0.5). |
| Observation placement | **Appended to `state`** | Also flows into `privileged_state` automatically via `jp.hstack([state, ...])`. |

### Resolution & Patch Size Analysis

Go1 physical constants (from `go1_mjx_feetonly.xml`):
- **Foot radius:** 0.023m → diameter = **4.6cm**
- **Stance width:** hip y-offset (4.675cm) + thigh y-offset (8cm) = **12.7cm from root to outer foot**

The patch must extend past the outer feet — not just to them — to give lookahead before a foot lands off the bridge.

| Patch | Half-coverage | Reaches feet (±12.7cm)? | Lookahead past feet |
|---|---|---|---|
| 7×7 @ 3cm | ±9cm | No — clips feet by 3.7cm | None |
| 9×9 @ 3cm | ±12cm | Barely (−0.7cm) | None |
| 11×11 @ 3cm | ±15cm | Yes (+2.3cm) | Minimal |
| **13×13 @ 3cm** | **±18cm** | **Yes (+5.3cm past feet)** | **~one step of lateral lookahead** |

**Recommendation: 3cm/cell, 13×13 patch (169 floats)**
- Foot (4.6cm) reliably spans 1–2 cells — edge presence is unambiguous
- 0.1m bridge ≈ 3 cells — enough to detect drift direction
- 0.3m bridge ≈ 10 cells — solid signal across the curriculum range
- ±18cm coverage sees 5.3cm past the outer feet — enough to detect and react before a foot lands off the bridge
- 5cm/cell ruled out: foot fits inside one cell, on/off-bridge status ambiguous at boundaries
- 2cm/cell ruled out: 13×13 patch shrinks to ±13cm, barely reaching the feet with no lookahead

---

## Implementation

All changes are in `mujoco_playground/_src/locomotion/go1/bridge.py`. No other files need
to change except the PPO config obs size note (see end).

### Step 1 — Build the world heightmap at `__init__` time

After the bridge-width patch in `__init__`, call `self._build_world_heightmap()` to store
`self._world_heightmap` as a JAX array.

Scene geometry (from `scene_mjx_feetonly_bridge.xml`):

| Geom | x span | y span | top surface z |
|---|---|---|---|
| `platform_a` | -3.0 to 0.0 | -1.0 to +1.0 | 0.5 |
| `bridge` | 0.0 to 4.0 | -`bridge_half_width` to +`bridge_half_width` | 0.5 |
| `platform_b` | 4.0 to 7.0 | -1.0 to +1.0 | 0.5 |

```python
# Constants — define at module level or as class attributes
_HM_X_MIN, _HM_X_MAX = -3.0, 7.0   # full scene x span
_HM_Y_MIN, _HM_Y_MAX = -1.0, 1.0   # full scene y span (platform width)
_HM_CELL = 0.03                      # 3 cm per cell
_HM_PATCH = 13                       # 13×13 patch

def _build_world_heightmap(self) -> None:
    nx = int(round((_HM_X_MAX - _HM_X_MIN) / _HM_CELL))  # 334
    ny = int(round((_HM_Y_MAX - _HM_Y_MIN) / _HM_CELL))  # 67

    xs = np.linspace(_HM_X_MIN, _HM_X_MAX, nx)
    ys = np.linspace(_HM_Y_MIN, _HM_Y_MAX, ny)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')  # (nx, ny) each

    hw = self._config.bridge_half_width
    on_platform_a = (xx >= -3.0) & (xx <= 0.0) & (np.abs(yy) <= 1.0)
    on_bridge      = (xx >=  0.0) & (xx <= 4.0) & (np.abs(yy) <= hw)
    on_platform_b  = (xx >=  4.0) & (xx <= 7.0) & (np.abs(yy) <= 1.0)

    world_heightmap = (on_platform_a | on_bridge | on_platform_b).astype(np.float32)
    self._world_heightmap = jp.array(world_heightmap)  # (334, 67) — fixed world grid
```

Also precompute the static local-frame offset grid once (avoids recomputing every step):

```python
# Also in _post_init() or __init__ after _build_world_heightmap()
half = (_HM_PATCH - 1) // 2  # = 6
idx = np.arange(_HM_PATCH) - half  # [-6, -5, ..., 0, ..., 6]
dx, dy = np.meshgrid(idx * _HM_CELL, idx * _HM_CELL, indexing='ij')
self._lh_dx = jp.array(dx)  # (13, 13) — forward offsets in robot-local frame
self._lh_dy = jp.array(dy)  # (13, 13) — lateral offsets in robot-local frame
```

Memory: 334 × 67 × 4 bytes ≈ **87 KB**. Negligible.

---

### Step 2 — Sample the local heightmap each step

Add a helper method `_get_local_heightmap(data)` that queries the world heightmap through
the robot's current position and yaw, returning a flat (169,) array in robot-local frame.

```python
def _get_local_heightmap(self, data: mjx.Data) -> jax.Array:
    # Robot position in world frame
    robot_x = data.qpos[0]
    robot_y = data.qpos[1]

    # Robot heading: forward vector in world frame from IMU site rotation matrix.
    # site_xmat column 0 = x-axis of site = forward direction in world frame.
    forward = data.site_xmat[self._imu_site_id] @ jp.array([1.0, 0.0, 0.0])
    cos_yaw = forward[0]
    sin_yaw = forward[1]

    # Rotate robot-local offsets into world frame so we query the right world cells.
    # self._lh_dx/dy are fixed offsets in robot-local frame (forward/left).
    # After rotation, dx_world/dy_world are the same offsets expressed in world frame.
    dx_world = cos_yaw * self._lh_dx - sin_yaw * self._lh_dy
    dy_world = sin_yaw * self._lh_dx + cos_yaw * self._lh_dy

    # Absolute world positions for each of the 13×13 sample points
    sx = robot_x + dx_world  # (13, 13)
    sy = robot_y + dy_world  # (13, 13)

    # Look up those world positions in the world heightmap
    nx, ny = self._world_heightmap.shape
    xi = jp.clip(jp.round((sx - _HM_X_MIN) / _HM_CELL).astype(jp.int32), 0, nx - 1)
    yi = jp.clip(jp.round((sy - _HM_Y_MIN) / _HM_CELL).astype(jp.int32), 0, ny - 1)

    # Result is 13×13 values arranged in robot-local frame — flatten for the obs vector
    local_heightmap = self._world_heightmap[xi, yi]  # (13, 13)
    return local_heightmap.ravel()                    # (169,)
```

Key details:
- `self._imu_site_id` is already set by the base class — same one used in `_cost_heading`.
- `jp.round(...).astype(jp.int32)` does nearest-neighbour sampling. No interpolation needed for binary maps.
- `jp.clip` handles boundary: cells outside the scene world bounds return the edge value (0.0 = void), which is correct — anything outside the defined terrain is void.

---

### Step 3 — Add to `_get_obs()`

```python
def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
    # ... existing noise sampling unchanged ...

    local_heightmap = self._get_local_heightmap(data)  # (169,) — robot-local frame

    state = jp.hstack([
        noisy_linvel,                             # 3
        noisy_gyro,                               # 3
        noisy_gravity,                            # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,                          # 12
        info["last_act"],                         # 12
        jp.array([x_progress]),                   # 1
        local_heightmap,                          # 169  ← new
    ])  # total: 215

    # privileged_state begins with state, so heightmap is included automatically.
    privileged_state = jp.hstack([
        state,
        # ... rest unchanged ...
    ])

    return {"state": state, "privileged_state": privileged_state}
```

**Observation size change:** 46 → **215** floats in `state`.

The heightmap patch is intentionally placed **last** in the vector. The existing 46 features
stay at the same indices, so any debugging comparisons against old checkpoints remain valid.

---

### Step 4 — Wire it into `__init__`

```python
def __init__(self, ...):
    super().__init__(...)
    # Existing bridge-width patch
    bridge_id = self._mj_model.geom("bridge").id
    self._mj_model.geom_size[bridge_id, 1] = self._config.bridge_half_width
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    self._build_world_heightmap()   # ← new, must come after bridge width is patched
    self._post_init()
```

`_build_world_heightmap()` must run **after** `bridge_half_width` is patched into the model,
since it reads `self._config.bridge_half_width` to define the narrow corridor.

---

## PPO Config

The PPO network input size is inferred automatically from the environment's observation shape
at training startup — no hardcoded size to update in `locomotion_params.py`.

However, confirm that the `network_factory` in the PPO config does not specify a fixed
`obs_size`. If it does, update it to 95. Check by grepping:

```sh
grep -n "obs_size\|215\|observation_size" mujoco_playground/config/locomotion_params.py
```

---

## Testing Before Training

On the Mac (no GPU), verify correctness without running the full training loop:

```python
import jax, jax.numpy as jp
from mujoco_playground import registry

env = registry.load("Go1BridgeCrossing")

# World heightmap checks — fixed grid, world coordinates
print(env._world_heightmap.shape)        # expect (334, 67)
print(env._world_heightmap.mean())       # expect ~0.4–0.6 (fraction of walkable cells)
print(env._world_heightmap[150, 33])     # x=1.5m (mid-bridge), y=0 (centre) → 1.0
print(env._world_heightmap[150, 0])      # x=1.5m, y=-1.0 (off bridge edge) → 0.0

# Local heightmap checks — robot-local frame, changes with position and yaw
rng = jax.random.PRNGKey(0)
state = env.reset(rng)
lh = env._get_local_heightmap(state.data)
print(lh.shape)              # (169,)
print(lh.reshape(13, 13))    # robot facing +x, centred on bridge → symmetric band of 1s

# Check obs size
print(state.obs["state"].shape)    # expect (215,)
```

Also visually verify the 13×13 local heightmap printout: robot centred on 0.8m bridge →
wide symmetric band of 1s. Robot centred on 0.1m bridge → only ~3 centre columns = 1.0.
Robot yawed 45° → diagonal band of 1s (the world heightmap never changes; only the local
view rotates).

---

## Curriculum Compatibility

The world heightmap is built once at `__init__` with `self._config.bridge_half_width`. Each
curriculum stage is a fresh training run with a new config override (e.g.
`--playground_config_overrides '{"bridge_half_width": 0.15}'`), so a new environment
instance is created and `_build_world_heightmap()` runs again with the correct width. No
special handling needed.

If a future continuous curriculum changes width per-episode (within a single run), the
heightmap would need to move into `reset()` and be passed through `info`. That is out of
scope for now.

---

## What This Does NOT Change

- Reward terms — no changes needed; the agent already has a lateral deviation penalty.
- XML files — no changes.
- Training command — same `train-jax-ppo` invocation. Start fresh from a new foundation run
  (0.8m) with the new obs so the policy learns to use the heightmap from the beginning.
- `randomize.py` — not used for bridge crossing; unchanged.
