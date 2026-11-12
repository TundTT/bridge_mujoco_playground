# Level walking with tail assist — reward draft (2026-11-12)

## Problem (results/posture/, scripts/diagnose_posture.py)
On narrow beams every tail-push policy stops putting its rear feet down and
walks on the front legs. Rear-foot contact is 0% at 0.06 m (active and no
tail), with the rear feet 16-24 cm in the air and the trunk pitched 37-49 deg
nose down. Front feet sit under the CoM, 10-17 cm behind the trunk centre. On
the platform all four feet step normally and pitch is 0-6 deg. The active tail
sits at about -60 deg (±10) as a sideways counterweight instead of swinging.

Why the reward allowed it (per-step values from the rollouts, progress ~30):
- foot_off_bridge (-50) is charged only at touchdown, so feet that never land never pay.
- Box bridge: feet_clearance and feet_height measured from the world floor, so
  every swing looked 0.5 m high; every touchdown cost ~7 before scaling and
  feet_clearance was -2.3 on any foot motion.
- The foot placement multiplier counts airborne feet as perfect when held over the centreline.
- Against it: feet_stale_air -3, orientation -0.4 to -2.9. Termination only at 60 deg tilt.

## Changes (all opt-in; defaults reproduce the old envs exactly)
bridge.py
- `foot_height_from_surface` (bool): swing-height terms measure from the 0.5 m surface.
- `pitch_terminate_deg` (0 = off): failure if |pitch| exceeds it.
- `reward_config.scales.pitch`, `.roll`: squared trunk pitch / roll (rad^2).

bridge_tail.py
- `reward_config.scales.tail_center`: tail angle^2 (stops parking the tail sideways).
- `reward_config.scales.tail_roll_assist`: max(tau * roll_rate, 0), capped at
  `tail_assist_cap` W. It rewards tail torque that opposes the trunk's roll rate. The sign was
  checked in sim: +7.9 Nm on the tail spins the trunk at -7.8 rad/s about x.

## Proposed preset
```json
{"foot_height_from_surface": true,
 "pitch_terminate_deg": 40.0,
 "reward_config.scales.pitch": -30.0,
 "reward_config.scales.roll": -10.0,
 "reward_config.scales.feet_stale_air": -20.0,
 "reward_config.scales.tail_center": -1.0,
 "reward_config.scales.tail_roll_assist": 0.1}
```
Estimated cost on the measured gaits (progress reward ~30/step):

| Gait | Pitch | Pitch cost | Stale-air now → new |
|---|---|---|---|
| Normal walking, platform / wide beam | 0-6 deg | -0.1 to -0.8 | 0 |
| 0.20 m beam | 6-10 deg | -0.4 to -1.1 | 0 to -0.6 |
| Front-legs-only, 0.06 m | 37-49 deg | -13 to -22 | -3 → -30 |

Under the preset the front-legs-only gait loses more per step than it earns,
and pitch over 40 deg ends the episode (-200). Normal walking pays under 1% of
the progress reward.

## Open choices
- tail_roll_assist is shaping and could be farmed by wobbling the trunk; the
  cap (10 W -> 1/step) and the roll/ang_vel costs bound it. Alternative: set it
  to 0 and rely on the level-trunk costs to make the tail worth using.
- Retraining needed: the old checkpoints are adapted to the two-leg gait.
  Suggest re-running the width curriculum from scratch for active + no tail.
