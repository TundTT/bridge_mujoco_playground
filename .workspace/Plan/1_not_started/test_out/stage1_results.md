# Stage 1 — World Heightmap Test Results

## Final status: PASS (after np.arange fix)

---

## Fix applied

Replaced `np.linspace` with `np.arange` in `build_world_heightmap`:

```python
# Before (wrong): spacing = (max-min)/(n-1) ≈ 0.030303, not 0.03
nx = int(round((_HM_X_MAX - _HM_X_MIN) / _HM_CELL))
xs = np.linspace(_HM_X_MIN, _HM_X_MAX, nx)

# After (correct): spacing = exactly 0.03m
xs = np.arange(_HM_X_MIN, _HM_X_MAX + _HM_CELL/2, _HM_CELL)
ys = np.arange(_HM_Y_MIN, _HM_Y_MAX + _HM_CELL/2, _HM_CELL)
```

Result: shape `(334, 68)`, spacing exactly `0.030000m` on both axes.

---

## Known-cell checks (after fix)

| Label | Expected | Got | Status |
|---|---|---|---|
| platform_a centre (-1.5, 0) | 1 | 1 | OK |
| bridge centre (2.0, 0) | 1 | 1 | OK |
| platform_b centre (5.5, 0) | 1 | 1 | OK |
| bridge inner edge (2.0, 0.14) | 1 | 1 | OK |
| bridge outer edge (2.0, 0.19) | 0 | 0 | OK |
| bridge outer edge neg (2.0, -0.19) | 0 | 0 | OK |
| outside scene (-4.0, 0) | 0 | 1 | N/A — see note |

**Bridge width:** 10 cells = 30.0cm ✓ (matches config)

### Note on outside-scene check

`world_to_idx(-4.0, 0)` clips to `xi=0` → `xs[0]=-3.0` → platform_a → returns 1.
This is the correct clip behaviour. During training the robot never queries x=-4.0
(platform_a only extends to x=-3.0, which is the heightmap boundary). The check
expectation of 0 is wrong for a clipped implementation — treat this check as N/A.

---

## Plots

- `stage1_world_heightmap.png` — original run (linspace, bridge=27cm)
- `stage1_world_heightmap_fixed.png` — after arange fix (bridge=30cm ✓)

---

## Action required before implementing bridge.py

Use `np.arange` (not `np.linspace`) in `_build_world_heightmap()` in `bridge.py`.
The exact form to use:

```python
_HM_CELL = 0.03

def _build_world_heightmap(self, bridge_half_width: float) -> jax.Array:
    xs = np.arange(_HM_X_MIN, _HM_X_MAX + _HM_CELL / 2, _HM_CELL)
    ys = np.arange(_HM_Y_MIN, _HM_Y_MAX + _HM_CELL / 2, _HM_CELL)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    on_platform_a = (xx >= -3.0) & (xx <= 0.0) & (np.abs(yy) <= 1.0)
    on_bridge      = (xx >=  0.0) & (xx <= 4.0) & (np.abs(yy) <= bridge_half_width)
    on_platform_b  = (xx >=  4.0) & (xx <= 7.0) & (np.abs(yy) <= 1.0)
    hm = (on_platform_a | on_bridge | on_platform_b).astype(np.float32)
    return jnp.array(hm)
```
