# Idea: True Inboard-Pull Quality Formulation

> Surfaced by Fable (Claude Opus 4.8) during sub-0.10m curriculum review, 2026-06-13.
> Not implemented — flagged for future use if pursuing sub-0.05m widths.

---

## The Problem with Current Quality

The current formulation:
```python
tau = jp.maximum(0.5 * info["virtual_hw"], 0.01)
quality = jp.exp(jp.minimum(margin, 0.0) / tau)
```

The `jp.minimum(margin, 0.0)` clip means **quality = 1.0 for every foot inside virtual_hw**, regardless of how close to the edge. There is zero gradient distinguishing "foot safely inboard" from "foot right at the edge." The reward only punishes feet that have already crossed outside the virtual boundary — it doesn't pull feet *away* from the edge.

This is fine for the 0.10m–0.06m range (the fading floor and foot_off_virtual provide enough signal), but at extreme widths where the policy needs a genuine safety margin, this formulation has a blind spot.

## The Fix

Replace the edge-referenced target with a target set inboard at half the virtual width:

```python
tau = jp.maximum(0.5 * info["virtual_hw"], 0.01)
target = 0.5 * info["virtual_hw"]          # aim for half-width inboard of edge
quality = jp.exp(-jp.maximum(jp.abs(foot_y) - target, 0.0) / tau)
```

**Behaviour:**
- `quality = 1.0` when foot is within `target = 0.5 * vhw` of centerline (comfortably inboard)
- Decays smoothly once foot moves outboard of `target`, with decay length `tau = 0.5 * vhw`
- At `|foot_y| = vhw` (edge): `quality = exp(-0.5*vhw / 0.5*vhw) = exp(-1.0) ≈ 0.37`
- Creates live gradient throughout `[target, vhw]` — the safety margin zone

**vs current:** current gives quality=1.0 from centerline all the way to the edge; this formulation gives quality=1.0 only in the inboard half, then decays.

## When to Use

- Sub-0.05m widths where roll stability requires feet to be genuinely centred, not merely inside the boundary
- Any stage where the policy appears to "hug the edge" and suffers falls despite high success rate
- If the morphological limit appears to be stability-driven (roll) rather than placement-driven (foot off bridge) — inboard pull helps with the former

## Trade-offs

- Changes the reward semantics for the wide stages too (quality degrades once foot crosses 0.5*vhw even on easy stages). This may slow learning early in the curriculum.
- Could be introduced only for narrow stages (e.g. vhw < 0.06) via a conditional, keeping wide-stage behaviour unchanged.
- The `target = 0.5 * vhw` choice is arbitrary — could tune to 0.6 or 0.7 of vhw.
