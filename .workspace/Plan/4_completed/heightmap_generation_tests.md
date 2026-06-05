# Heightmap Generation Tests

> Tests to run **before** any training to confirm the world heightmap and local heightmap
> are generated correctly. All tests run on Mac (no GPU, no full training stack needed).
> Three stages. Stage 1: numpy only, no env. Stage 2: ground truth check against MuJoCo model (run after implementing bridge.py). Stage 3: visual local heightmap check.
>
> **Status: ALL THREE STAGES PASSED — 2026-06-04**
>
> Key finding during Stage 1: the original test script used `np.linspace` which gives
> spacing ≈ 0.030303m (not exactly 0.03m), making the bridge 27cm instead of 30cm.
> Fixed by switching to `np.arange` in both the test and in `bridge.py`.
> See `test_out/stage1_results.md`, `test_out/stage2_results.md`, `test_out/stage3_results.md`
> for full details and plots.

---

## Stage 1 — World Heightmap (numpy only, no env needed) ✓ PASSED

This tests the generation logic in isolation — before wiring it into `bridge.py`.
Copy this script and run it directly.

> **Fix applied:** Replace `np.linspace` with `np.arange(_HM_X_MIN, _HM_X_MAX + _HM_CELL/2, _HM_CELL)`
> (and same for Y). Linspace with N=333 gives spacing 0.030303m not 0.03m, causing the bridge
> to be 27cm instead of 30cm. Arange gives exact 0.03m spacing → shape (334, 68), bridge = 30cm ✓.
> Also add bounds clipping to `world_to_idx` to prevent numpy negative-index wrapping on
> out-of-bounds queries. The test script below reflects these fixes.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# === Paste the generation logic here (mirrors what will go in bridge.py) ===

_HM_X_MIN, _HM_X_MAX = -3.0, 7.0
_HM_Y_MIN, _HM_Y_MAX = -1.0, 1.0
_HM_CELL  = 0.03
BRIDGE_HALF_WIDTH = 0.15   # ← change this to test different curriculum widths

def build_world_heightmap(bridge_half_width):
    # Use arange (not linspace) for exact 0.03m spacing
    xs = np.arange(_HM_X_MIN, _HM_X_MAX + _HM_CELL/2, _HM_CELL)
    ys = np.arange(_HM_Y_MIN, _HM_Y_MAX + _HM_CELL/2, _HM_CELL)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    on_platform_a = (xx >= -3.0) & (xx <= 0.0) & (np.abs(yy) <= 1.0)
    on_bridge      = (xx >=  0.0) & (xx <= 4.0) & (np.abs(yy) <= bridge_half_width)
    on_platform_b  = (xx >=  4.0) & (xx <= 7.0) & (np.abs(yy) <= 1.0)
    return (on_platform_a | on_bridge | on_platform_b).astype(np.float32)

hm = build_world_heightmap(BRIDGE_HALF_WIDTH)
nx, ny = hm.shape

# ── Check 1: shape ────────────────────────────────────────────────────────────
print(f"Shape: {hm.shape}")   # expect (334, 68)

# ── Check 2: known good cells ─────────────────────────────────────────────────
def world_to_idx(x, y):
    xi = int(round((x - _HM_X_MIN) / _HM_CELL))
    yi = int(round((y - _HM_Y_MIN) / _HM_CELL))
    # Clip to prevent numpy negative-index wrapping on out-of-bounds queries
    return np.clip(xi, 0, nx - 1), np.clip(yi, 0, ny - 1)

checks = [
    ((-1.5,  0.0), 1.0, "platform_a centre"),
    (( 2.0,  0.0), 1.0, "bridge centre"),
    (( 5.5,  0.0), 1.0, "platform_b centre"),
    (( 2.0,  BRIDGE_HALF_WIDTH - 0.01), 1.0, "bridge inner edge"),
    (( 2.0,  BRIDGE_HALF_WIDTH + 0.04), 0.0, "bridge outer edge (void)"),
    (( 2.0, -BRIDGE_HALF_WIDTH - 0.04), 0.0, "bridge outer edge neg (void)"),
    ((-4.0,  0.0), 0.0, "outside scene bounds (void)"),
]
print("\nKnown-cell checks:")
all_ok = True
for (x, y), expected, label in checks:
    xi, yi = world_to_idx(x, y)
    got = hm[xi, yi]
    status = "OK" if got == expected else "FAIL"
    if got != expected:
        all_ok = False
    print(f"  [{status}] ({x:.2f}, {y:.3f}) {label}: got {got:.0f}, expected {expected:.0f}")
print("All checks passed!" if all_ok else "SOME CHECKS FAILED — inspect the plot.")

# ── Check 3: bridge width in cells ───────────────────────────────────────────
bridge_col_centre = int(round((2.0 - _HM_X_MIN) / _HM_CELL))
bridge_row = hm[bridge_col_centre, :]   # lateral slice at x=2m
n_bridge_cells = int(bridge_row.sum())
expected_cells = int(round(2 * BRIDGE_HALF_WIDTH / _HM_CELL))
print(f"\nBridge width: {n_bridge_cells} cells (expected ~{expected_cells})")
print(f"  = {n_bridge_cells * _HM_CELL * 100:.1f} cm  (config: {2*BRIDGE_HALF_WIDTH*100:.0f} cm)")

# ── Visual: full world heightmap ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Top-down view (transpose so x=horizontal, y=vertical)
ax = axes[0]
ax.imshow(hm.T, origin='lower',
          extent=[_HM_X_MIN, _HM_X_MAX, _HM_Y_MIN, _HM_Y_MAX],
          cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_title(f"World Heightmap — bridge_half_width={BRIDGE_HALF_WIDTH}m")
ax.set_xlabel("World X (m)")
ax.set_ylabel("World Y (m)")
ax.axvline(0, color='blue', lw=0.8, ls='--', label='bridge start')
ax.axvline(4, color='blue', lw=0.8, ls='--', label='bridge end')
ax.axhline( BRIDGE_HALF_WIDTH, color='orange', lw=0.8, ls=':')
ax.axhline(-BRIDGE_HALF_WIDTH, color='orange', lw=0.8, ls=':', label='bridge edge')
ax.legend(fontsize=8)

# Lateral slice at x=2m (mid-bridge) — should show bridge width clearly
ax = axes[1]
ys = np.linspace(_HM_Y_MIN, _HM_Y_MAX, hm.shape[1])
ax.plot(ys, hm[bridge_col_centre, :], 'k-', lw=2)
ax.axvline( BRIDGE_HALF_WIDTH, color='orange', ls='--', label='bridge edge')
ax.axvline(-BRIDGE_HALF_WIDTH, color='orange', ls='--')
ax.set_title("Lateral slice at x=2m (mid-bridge)")
ax.set_xlabel("World Y (m)")
ax.set_ylabel("Value (1=bridge, 0=void)")
ax.set_ylim(-0.1, 1.1)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/world_heightmap.png", dpi=150)
plt.show()
print("\nPlot saved to /tmp/world_heightmap.png")
```

### What to look for in the plot

**Top-down view:**
- Green = bridge surface, Red = void
- Two large green platforms connected by a narrow green corridor
- The corridor width visually matches `bridge_half_width`
- No gaps or steps at x=0 and x=4 (platform-to-bridge seams)

**Lateral slice:**
- Flat 1.0 for exactly the bridge width, then drops to 0.0
- Orange lines (bridge edges from config) should align with the drop

---

## Stage 2 — Ground Truth Check (requires JAX + env, run after implementing bridge.py) ✓ PASSED

> **Result: 0 / 22,712 cells disagree (0.00%) — perfect match.**
> Geom bounds confirmed: platform_a x∈[-3,0], bridge x∈[0,4] y∈[-0.4,0.4], platform_b x∈[4,7].

This is the critical test: it compares `self._world_heightmap` (what our code computed)
against `env.mj_model` (what MuJoCo actually loaded from the XML). Any mismatch means
the heightmap is lying to the policy about where the bridge is.

The ground truth is read directly from the loaded model's geom parameters — position and
half-extents — rather than from our hardcoded constants. This catches cases where the two
disagree (e.g. bridge was width-patched in the model but the heightmap was built before the
patch, or our x/y bounds don't exactly match the geom footprints).

```python
import numpy as np
import matplotlib.pyplot as plt
from mujoco_playground import registry

env = registry.load("Go1BridgeCrossing")

# ── Read ACTUAL geom parameters from the loaded MuJoCo model ─────────────────
# These are ground truth — whatever the physics engine is actually using.
def geom_bounds(model, name):
    """Return (x_min, x_max, y_min, y_max) for a box geom by name."""
    gid = model.geom(name).id
    px, py, _ = model.geom_pos[gid]
    sx, sy, _ = model.geom_size[gid]
    return px - sx, px + sx, py - sy, py + sy

pa_x0, pa_x1, pa_y0, pa_y1 = geom_bounds(env.mj_model, "platform_a")
br_x0, br_x1, br_y0, br_y1 = geom_bounds(env.mj_model, "bridge")
pb_x0, pb_x1, pb_y0, pb_y1 = geom_bounds(env.mj_model, "platform_b")

print("Ground truth geom bounds from MuJoCo model:")
print(f"  platform_a: x=[{pa_x0}, {pa_x1}]  y=[{pa_y0}, {pa_y1}]")
print(f"  bridge:     x=[{br_x0}, {br_x1}]  y=[{br_y0}, {br_y1}]")
print(f"  platform_b: x=[{pb_x0}, {pb_x1}]  y=[{pb_y0}, {pb_y1}]")
print(f"  bridge_half_width from config: {env._config.bridge_half_width}")
print(f"  bridge_half_width from model:  {env.mj_model.geom_size[env.mj_model.geom('bridge').id, 1]}")

# ── Build the MuJoCo ground-truth map ────────────────────────────────────────
# Same grid as the world heightmap, but values derived from the actual model.
_HM_X_MIN, _HM_X_MAX = -3.0, 7.0
_HM_Y_MIN, _HM_Y_MAX = -1.0, 1.0
_HM_CELL = 0.03

xs = np.arange(_HM_X_MIN, _HM_X_MAX + _HM_CELL/2, _HM_CELL)
ys = np.arange(_HM_Y_MIN, _HM_Y_MAX + _HM_CELL/2, _HM_CELL)
xx, yy = np.meshgrid(xs, ys, indexing='ij')

on_platform_a = (xx >= pa_x0) & (xx <= pa_x1) & (yy >= pa_y0) & (yy <= pa_y1)
on_bridge      = (xx >= br_x0) & (xx <= br_x1) & (yy >= br_y0) & (yy <= br_y1)
on_platform_b  = (xx >= pb_x0) & (xx <= pb_x1) & (yy >= pb_y0) & (yy <= pb_y1)
ground_truth = (on_platform_a | on_bridge | on_platform_b).astype(np.float32)

# ── Compare against our world heightmap ──────────────────────────────────────
our_hm = np.array(env._world_heightmap)
diff = our_hm - ground_truth          # 0 = agree, +1 = we say bridge / MuJoCo says void
                                      #             -1 = we say void  / MuJoCo says bridge
n_disagree = int((diff != 0).sum())
n_total = len(xs) * len(ys)

print(f"\nComparison: {n_disagree} / {n_total} cells disagree "
      f"({100 * n_disagree / n_total:.2f}%)")

if n_disagree == 0:
    print("PASS — world heightmap matches MuJoCo model exactly.")
else:
    false_bridge = int((diff > 0).sum())   # heightmap says bridge, model says void
    false_void   = int((diff < 0).sum())   # heightmap says void,   model says bridge
    print(f"FAIL:")
    print(f"  {false_bridge} cells: heightmap says BRIDGE, MuJoCo says VOID")
    print(f"  {false_void}   cells: heightmap says VOID,   MuJoCo says BRIDGE")

# ── Visual diff ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

extent = [_HM_X_MIN, _HM_X_MAX, _HM_Y_MIN, _HM_Y_MAX]

axes[0].imshow(our_hm.T, origin='lower', extent=extent,
               cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
axes[0].set_title("Our world heightmap")

axes[1].imshow(ground_truth.T, origin='lower', extent=extent,
               cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
axes[1].set_title("MuJoCo ground truth")

# Difference map: white=agree, red=we say bridge/MuJoCo void, blue=we say void/MuJoCo bridge
diff_plot = axes[2].imshow(diff.T, origin='lower', extent=extent,
                            cmap='bwr', vmin=-1, vmax=1, aspect='auto')
axes[2].set_title(f"Difference ({n_disagree} cells)")
plt.colorbar(diff_plot, ax=axes[2])

for ax in axes:
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")

plt.tight_layout()
plt.savefig("/tmp/heightmap_ground_truth_diff.png", dpi=150)
plt.show()
print("Diff plot saved to /tmp/heightmap_ground_truth_diff.png")
```

### What to look for

- **Difference plot all white** = perfect match. Done.
- **Red cells** = our heightmap marks these as bridge but MuJoCo doesn't. The policy thinks it can stand here but it will fall through.
- **Blue cells** = our heightmap marks these as void but MuJoCo says it's surface. The policy avoids a safe area.
- **Red/blue at the bridge edges only, 1–2 cells wide** = acceptable boundary rounding. The 3cm grid can't represent a perfectly sharp edge; ±3cm of rounding at the transition is fine.
- **Red/blue spanning the whole bridge width** = the bridge half-width in the heightmap doesn't match the model — almost certainly means `_build_world_heightmap()` ran before the width patch.

---

## Stage 3 — Local Heightmap (requires JAX + env) ✓ PASSED

> **Result: all 5 cases match physical expectations.**
> Load with `config_overrides={"bridge_half_width": 0.15}` to see edge structure — the default
> 0.8m bridge fills the entire 13×13 patch (±18cm coverage can't reach ±40cm edges).
> A 1-cell asymmetry in the centred case is expected: the arange grid at Y doesn't include
> y=±0.15 exactly, so the negative edge rounds 1 cell further out than the positive edge. Fine.

Run from inside `bridge_mujoco_playground/` with the venv active.

```python
import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from mujoco_playground import registry
from mujoco.mjx._src import math

env = registry.load("Go1BridgeCrossing")
rng = jax.random.PRNGKey(0)
state = env.reset(rng)

# Helper: manually set robot pose and call _get_local_heightmap
def get_patch_at(env, x, y, yaw_deg):
    """Reset the env, teleport the robot, return the 13x13 local heightmap."""
    import mujoco
    yaw = np.deg2rad(yaw_deg)
    data = state.data

    # Patch qpos: position + quaternion
    qpos = np.array(env._init_q)
    qpos[0] = x
    qpos[1] = y
    quat = np.array(math.axis_angle_to_quat(jp.array([0., 0., 1.]), jp.array([yaw])))
    qpos[3:7] = quat

    from mujoco import mjx
    from mujoco_playground._src import mjx_env
    data = mjx_env.make_data(env.mj_model, qpos=jp.array(qpos),
                              qvel=jp.zeros(env.mjx_model.nv),
                              ctrl=jp.array(qpos[7:]),
                              impl=env.mjx_model.impl.value,
                              naconmax=env._config.naconmax,
                              njmax=env._config.njmax)
    data = mjx.forward(env.mjx_model, data)
    lh = env._get_local_heightmap(data)
    return np.array(lh).reshape(13, 13)

# ── Test cases ────────────────────────────────────────────────────────────────
cases = [
    (2.0,  0.00,   0, "Mid-bridge, centred, facing +x"),
    (2.0,  0.00,  45, "Mid-bridge, centred, yawed 45° left"),
    (2.0,  0.10,   0, "Mid-bridge, drifted right, facing +x"),
    (2.0,  0.00,  90, "Mid-bridge, centred, facing sideways (+y)"),
    (-1.5, 0.00,   0, "Platform A, facing +x"),
]

fig, axes = plt.subplots(1, len(cases), figsize=(4 * len(cases), 4))

for ax, (x, y, yaw, title) in zip(axes, cases):
    patch = get_patch_at(env, x, y, yaw)
    im = ax.imshow(patch, origin='upper', cmap='RdYlGn', vmin=0, vmax=1)
    # Mark robot centre
    ax.plot(6, 6, 'b*', markersize=12, label='robot')
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("← LEFT   RIGHT →", fontsize=7)
    ax.set_ylabel("BEHIND ↑  ↓ AHEAD", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Local Heightmap (13×13) — green=bridge, red=void, ★=robot", fontsize=10)
plt.tight_layout()
plt.savefig("/tmp/local_heightmap_tests.png", dpi=150)
plt.show()
print("Plot saved to /tmp/local_heightmap_tests.png")
```

### What to look for in the plot

| Test case | Expected pattern |
|---|---|
| Mid-bridge, centred, facing +x | Symmetric vertical green band, red on both sides |
| Mid-bridge, centred, yawed 45° left | Green diagonal band top-left → bottom-right |
| Mid-bridge, drifted right, facing +x | Green band shifted left — more red on the right |
| Mid-bridge, facing sideways (+y) | Green horizontal band (bridge now runs left-right in the patch) |
| Platform A, facing +x | All green (robot is on the wide platform) |

If any case looks wrong — symmetric when it should be diagonal, or all-void when it should
be all-green — the bug is in either the yaw rotation math or the world coordinate lookup.

---

## Quick ASCII Sanity Check (no matplotlib needed)

If you want a fast check without plotting:

```python
def print_patch(patch, label):
    print(f"\n{label}")
    print("  ←LEFT  RIGHT→")
    for i, row in enumerate(patch):
        prefix = "AHEAD " if i == 0 else "HERE  " if i == 6 else "BEHIND" if i == 12 else "      "
        print(prefix + " " + " ".join("#" if v > 0.5 else "." for v in row))

print_patch(get_patch_at(env, 2.0, 0.0,  0), "Facing +x, centred")
print_patch(get_patch_at(env, 2.0, 0.0, 45), "Yawed 45° left")
```

Actual output for 0.3m bridge (half-width=0.15), verified 2026-06-04:

```
Facing +x, centred
  ←LEFT  RIGHT→
AHEAD  . . # # # # # # # # # # .
       . . # # # # # # # # # # .
       . . # # # # # # # # # # .
       . . # # # # # # # # # # .
       . . # # # # # # # # # # .
       . . # # # # # # # # # # .
HERE   . . # # # # # # # # # # .
       . . # # # # # # # # # # .
       . . # # # # # # # # # # .
       . . # # # # # # # # # # .
       . . # # # # # # # # # # .
       . . # # # # # # # # # # .
BEHIND . . # # # # # # # # # # .

Yawed 45° left
  ←LEFT  RIGHT→
AHEAD  . . . . . . # # # # # # #
       . . . . . # # # # # # # #
       . . . . # # # # # # # # #
       . . . # # # # # # # # # #
       . . # # # # # # # # # # #
       . # # # # # # # # # # # #
HERE   # # # # # # # # # # # # #
       # # # # # # # # # # # # #
       # # # # # # # # # # # # .
       # # # # # # # # # # # . .
       # # # # # # # # # # . . .
       # # # # # # # # # . . . .
BEHIND # # # # # # # # . . . . .
```

Note: the centred case has 2 void columns on the left, 1 on the right (not 1 each side as
originally predicted). This is a ±1 cell grid-rounding artifact, not a bug — see Stage 3
notes above. The 45° case diagonal direction is mirrored vs the original prediction; this is
a label convention issue (column 0 = local −y = robot's right, not left). The rotation is
physically correct.
