# Plan: Local Heightmap Observation for Bridge Edge Perception

> **Status: NOT STARTED**
>
> Addresses the perception wall hit at 0.3m bridge width. The policy currently has no signal
> for where the bridge edges are — this adds a local terrain patch to the observation so the
> robot can sense drift and self-correct before falling off.

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
3. Add the flattened patch (49 floats) to the policy's `state` observation vector.

The robot's position within the patch immediately encodes "edge is N cells to the left/right",
giving it the reactive centering reflex that's currently missing.

This was suggested as a "world depth map + robot position in it" — the heightmap is the
depth map of the world, and indexing into it by robot position is how the robot knows where
it sits within that map.

---

## Why This Works in MJX

- The bridge geometry is **static within an episode** — no per-step rendering needed.
- The heightmap is a pre-computed JAX array stored as `self._heightmap`; sampling it is just array indexing.
- Array indexing is fully JIT-compilable and vectorises across the environment batch with no overhead.
- The heightmap only needs to be recomputed when `bridge_half_width` changes (i.e. at the start of a curriculum stage, not per step).

This avoids the depth-camera rendering problem entirely: there is no camera, no image, no
GPU rendering pipeline. Just a lookup table and an index.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Patch size | **7×7** | Wider coverage than 5×5; see resolution analysis. |
| Grid resolution | **3cm/cell** | Balances foot resolution vs. patch coverage; see analysis. |
| Coordinate frame | **Robot-local (rotated by yaw)** | Policy sees same spatial pattern regardless of heading. |
| Height encoding | **Binary (1.0 = surface, 0.0 = void)** | Simplest; all terrain surfaces are at the same height (z=0.5). |
| Observation placement | **Appended to `state`** | Also flows into `privileged_state` automatically via `jp.hstack([state, ...])`. |

### Resolution & Patch Size Analysis

Go1 physical constants (from `go1_mjx_feetonly.xml`):
- **Foot radius:** 0.023m → diameter = **4.6cm**
- **Stance width:** hip y-offset (4.675cm) + thigh y-offset (8cm) = 12.675cm per side → **~25cm foot-to-foot**

Resolution tradeoffs against these numbers:

| Resolution | Foot size in cells | 0.1m bridge | 0.3m bridge | 7×7 patch coverage |
|---|---|---|---|---|
| 5cm/cell | 0.9 cells (sub-cell, ambiguous at edges) | 2 cells (barely L vs R) | 6 cells | 35cm × 35cm |
| **3cm/cell** | **1.5 cells (straddles but detectable)** | **~3 cells** | **10 cells** | **21cm × 21cm** |
| 2cm/cell | 2.3 cells (reliable) | 5 cells | 15 cells | 14cm × 14cm (too narrow) |

**Recommendation: 3cm/cell, 7×7 patch (49 floats)**
- Foot (4.6cm) reliably spans 1–2 cells — edge presence is unambiguous
- 0.1m bridge = ~3 cells — enough to detect drift direction
- 0.3m bridge = 10 cells — solid gradient for the curriculum range where most training happens
- 21cm × 21cm coverage sees ~8cm past the outer feet on each side — enough lookahead for a gait cycle to respond
- 5cm/cell was ruled out because the foot fits inside a single cell, making on/off-bridge status ambiguous at boundaries
- 2cm/cell was ruled out because the 7×7 patch shrinks to 14cm × 14cm, barely covering the robot's own stance width with no lookahead

---

## Implementation

All changes are in `mujoco_playground/_src/locomotion/go1/bridge.py`. No other files need
to change except the PPO config obs size note (see end).

### Step 1 — Build the heightmap at `__init__` time

After the bridge-width patch in `__init__`, call `self._build_heightmap()` to store
`self._heightmap` as a JAX array.

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
_HM_PATCH = 7                        # 7×7 patch

def _build_heightmap(self) -> None:
    nx = int(round((_HM_X_MAX - _HM_X_MIN) / _HM_CELL))  # 334
    ny = int(round((_HM_Y_MAX - _HM_Y_MIN) / _HM_CELL))  # 67

    xs = np.linspace(_HM_X_MIN, _HM_X_MAX, nx)
    ys = np.linspace(_HM_Y_MIN, _HM_Y_MAX, ny)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')  # (nx, ny) each

    hw = self._config.bridge_half_width
    on_platform_a = (xx >= -3.0) & (xx <= 0.0) & (np.abs(yy) <= 1.0)
    on_bridge      = (xx >=  0.0) & (xx <= 4.0) & (np.abs(yy) <= hw)
    on_platform_b  = (xx >=  4.0) & (xx <= 7.0) & (np.abs(yy) <= 1.0)

    heightmap = (on_platform_a | on_bridge | on_platform_b).astype(np.float32)
    self._heightmap = jp.array(heightmap)  # (334, 67), stored on device
```

Also precompute the static local-frame offset grid once (avoids recomputing every step):

```python
# Also in _post_init() or __init__ after _build_heightmap()
half = (_HM_PATCH - 1) // 2  # = 3
idx = np.arange(_HM_PATCH) - half  # [-3, -2, -1, 0, 1, 2, 3]
dx, dy = np.meshgrid(idx * _HM_CELL, idx * _HM_CELL, indexing='ij')
self._patch_dx = jp.array(dx)  # (7, 7) — forward offsets in robot frame
self._patch_dy = jp.array(dy)  # (7, 7) — lateral offsets in robot frame
```

Memory: 334 × 67 × 4 bytes ≈ **87 KB**. Negligible.

---

### Step 2 — Sample the local patch in `_get_obs()`

Add a helper method `_sample_heightmap(data)` that returns a flat (49,) array.

```python
def _sample_heightmap(self, data: mjx.Data) -> jax.Array:
    # Robot world position
    robot_x = data.qpos[0]
    robot_y = data.qpos[1]

    # Robot heading: forward vector in world frame from IMU site rotation matrix.
    # site_xmat row 0 = x-axis of site = forward direction.
    forward = data.site_xmat[self._imu_site_id] @ jp.array([1.0, 0.0, 0.0])
    cos_yaw = forward[0]
    sin_yaw = forward[1]

    # Rotate local offsets into world frame.
    # (robot-local x = forward, y = left)
    dx_world = cos_yaw * self._patch_dx - sin_yaw * self._patch_dy
    dy_world = sin_yaw * self._patch_dx + cos_yaw * self._patch_dy

    # World-frame sample positions
    sx = robot_x + dx_world  # (7, 7)
    sy = robot_y + dy_world  # (7, 7)

    # Convert to integer grid indices, clamped to valid range
    nx, ny = self._heightmap.shape
    xi = jp.clip(jp.round((sx - _HM_X_MIN) / _HM_CELL).astype(jp.int32), 0, nx - 1)
    yi = jp.clip(jp.round((sy - _HM_Y_MIN) / _HM_CELL).astype(jp.int32), 0, ny - 1)

    # Gather values and flatten
    patch = self._heightmap[xi, yi]  # (7, 7)
    return patch.ravel()             # (49,)
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

    heightmap_patch = self._sample_heightmap(data)  # (49,)

    state = jp.hstack([
        noisy_linvel,                             # 3
        noisy_gyro,                               # 3
        noisy_gravity,                            # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,                          # 12
        info["last_act"],                         # 12
        jp.array([x_progress]),                   # 1
        heightmap_patch,                          # 49  ← new
    ])  # total: 95

    # privileged_state begins with state, so heightmap is included automatically.
    privileged_state = jp.hstack([
        state,
        # ... rest unchanged ...
    ])

    return {"state": state, "privileged_state": privileged_state}
```

**Observation size change:** 46 → **95** floats in `state`.

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

    self._build_heightmap()   # ← new, must come after bridge width is patched
    self._post_init()
```

`_build_heightmap()` must run **after** `bridge_half_width` is patched into the model,
since it reads `self._config.bridge_half_width` to define the narrow corridor.

---

## PPO Config

The PPO network input size is inferred automatically from the environment's observation shape
at training startup — no hardcoded size to update in `locomotion_params.py`.

However, confirm that the `network_factory` in the PPO config does not specify a fixed
`obs_size`. If it does, update it to 95. Check by grepping:

```sh
grep -n "obs_size\|46\|observation_size" mujoco_playground/config/locomotion_params.py
```

---

## Testing Before Training

On the Mac (no GPU), verify correctness without running the full training loop:

```python
import jax, jax.numpy as jp
from mujoco_playground import registry

env = registry.load("Go1BridgeCrossing")

# Check heightmap shape and values
print(env._heightmap.shape)        # expect (334, 67)
print(env._heightmap.mean())       # expect ~0.4–0.6 (fraction of walkable cells)
print(env._heightmap[150, 33])     # x=1.5m (mid-bridge), y=0 (centre) → 1.0

# Check patch sampling
rng = jax.random.PRNGKey(0)
state = env.reset(rng)
patch = env._sample_heightmap(state.data)
print(patch.shape)   # (49,)
print(patch.reshape(7, 7))  # should show bridge (1s) in centre rows, void (0s) on sides

# Check obs size
print(state.obs["state"].shape)    # expect (95,)
```

Also visually verify the 7×7 patch printout: when the robot is centred on a 0.4m bridge,
the middle 3–4 columns of the patch should be 1.0 and the outer columns 0.0 (void).
If the bridge is 0.1m wide, only the centre ~1 column should be 1.0.

---

## Curriculum Compatibility

The heightmap is built once at `__init__` with `self._config.bridge_half_width`. Each
curriculum stage is a fresh training run with a new config override (e.g.
`--playground_config_overrides '{"bridge_half_width": 0.15}'`), so a new environment
instance is created and `_build_heightmap()` runs again with the correct width. No special
handling needed.

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
