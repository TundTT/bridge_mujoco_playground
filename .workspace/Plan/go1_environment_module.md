# Phase: Go1 Bridge Environment Module

> **Status: READY TO IMPLEMENT**
>
> The Go1 bridge XML (`go1/xmls/scene_mjx_feetonly_bridge.xml`) is already complete and visually verified (see `bridge_terrain_xml.md`). The remaining work is creating `go1/bridge.py` and registering it. No new XML, constants file, or base class is needed — Go1 already has all of those.

---

## What Already Exists

Go1 has a complete scaffolding:
- `go1_constants.py` — sensor names, body names, XML paths
- `base.py` — `Go1Env` base class with all sensor helpers
- `go1_mjx_feetonly.xml` — robot model with feetonly collision and full sensor set
- `scene_mjx_feetonly_bridge.xml` — the bridge terrain scene (complete, verified)
- `sensor_bridge_feet.xml` — contact sensors for each foot × {platform_a, bridge, platform_b}

The only file to create is `go1/bridge.py`.

---

## What `go1/bridge.py` Needs to Do

### Class structure

`BridgeCrossing(go1_base.Go1Env)` — subclasses the existing `Go1Env` base class exactly as `Joystick` does, but with different config, reset, step, and reward logic.

### `default_config()`

Same pattern as `go1/joystick.py` but with bridge-specific fields added:

- `bridge_half_width` — the curriculum dial. Defaults to `0.2` (0.4m full width, medium difficulty). At `__init__` time this value patches the `bridge` geom's y half-extent in-place, then re-serialises the MJX model. No XML reload needed.
- `fall_threshold` — used in height termination. Defaults to `0.2`. This is a named config field so it can be tuned without touching code.
- `impl` — default to `"warp"` (consistent with joystick.py). Use `"jax"` if no GPU is available.

All other fields (Kp, Kd, ctrl_dt, sim_dt, noise_config, etc.) can be copied from joystick.py defaults unchanged.

---

## Termination Logic

Two conditions combined with `|`:

1. **Tilt termination** — `upvector[-1] < 0.0`. The robot's z-axis has rotated past horizontal. Already used in `joystick.py:306`. Catches falls on the bridge surface where the robot tips over.

2. **Height termination** — `data.qpos[2] < config.fall_threshold`. Catches the case where the robot walks off the edge of a platform or the bridge end and is in free-fall. Tilt alone does not fire fast enough for this case because the robot remains upright during the initial fall.

`fall_threshold = 0.2` means the episode ends when the root body drops to 0.2m — which is 0.3m below the 0.5m platform surface. This value is a starting guess; it should be swept in early training runs. Too high triggers spurious truncations when the robot leans over the edge; too low wastes training steps on a long free-fall.

---

## Bridge Width Curriculum

At `__init__` time, after the base class loads the model:
1. Look up the `bridge` geom by name.
2. Overwrite its y half-extent with `config.bridge_half_width`.
3. Re-run `mjx.put_model` to push the patched model to the MJX backend.

This keeps the XML static. The curriculum changes difficulty by passing `config_overrides={"bridge_half_width": 0.05}` at load time, which narrows the bridge from 0.4m down to 0.1m without any file changes.

---

## Reset Randomisation

Spawn on Platform A with:
- Small x/y position jitter (e.g. ±0.3m) — keeps the robot within the platform footprint but prevents overfitting to a single spawn point.
- Small yaw jitter (e.g. ±0.5 rad, ~30°) — forces the policy to learn to correct its heading rather than always approaching the bridge perfectly aligned.

No velocity noise at reset (unlike joystick.py which randomises initial velocity).

---

## Contact Detection

The bridge scene has no single floor geom. Go1's `base.py` initialises `_feet_floor_found_sensor` pointing to `{geom}_floor_found` sensors — but those sensors reference a `floor` geom that does not exist in the bridge scene, so bridge.py must not call or rely on them.

Instead, bridge.py initialises its own sensor address array covering all 12 foot × terrain combinations from `sensor_bridge_feet.xml`. At each step, the four per-foot contact signals are computed by OR-ing all three terrain contacts for that foot. This gives a single boolean per foot: "is this foot grounded anywhere?"

---

## Reward Design

This is a **goal-directed** task (reach Platform B), not a joystick command-following task. The reward structure is different from joystick.py:

| Term | Sign | Purpose |
|---|---|---|
| `forward_vel` | + | Reward clipped forward (x-axis) velocity. Drives the robot toward the bridge. |
| `goal_reached` | + | **[TO ADD]** One-time sparse bonus when the robot's x position crosses into Platform B. Without this, the robot has no explicit signal that reaching Platform B is the goal — it only knows moving forward is good. Keep `forward_vel` for dense learning signal but pair it with this for task completion. |
| `orientation` | − | Penalise non-upright tilt (`upvector[:2]` squared sum). Keeps the robot stable on the bridge. |
| `alive` | + | Small constant bonus per step. Ensures the agent prefers surviving over dying quickly. Counteracts the termination penalty at the margin. |
| `termination` | − | One-shot penalty when done=True. Discourages falls. |
| `torques` | − | Penalise large actuator forces. Efficiency regulariser. |
| `action_rate` | − | Penalise large changes between consecutive actions. Smoothness regulariser. |
| `energy` | − | Penalise torque × velocity. Efficiency regulariser. |
| `feet_air_time` | + | Reward feet that have been airborne and just made contact. Encourages a proper gait rather than shuffling. |
| `lateral_deviation` | − | **[TO ADD]** Penalise COM drifting from the bridge centreline (`-data.qpos[1]²`). Without this the robot has no incentive to stay centred and may learn to hug one edge all the way across, which fails on narrow widths. |

There is no command vector and no tracking reward — the robot's only goal is to move forward and stay upright.

---

## Observation Design

**Policy obs:** local linear velocity (3), gyro (3), gravity vector (3), joint angles relative to default pose (12), joint velocities (12), last action (12), x-progress (1). Total: 46 dimensions.

The `x-progress` signal is root x normalised to roughly [−1, 1] over the 7m terrain span. It gives the policy its position along the task without full odometry.

**Privileged obs:** everything in policy obs plus noiseless IMU signals, actuator forces, foot contact flags, foot velocities, and raw x position. Used by the critic in asymmetric actor-critic training.

Observation noise uses the same scale factors as joystick.py.

---

## Registration

Add `"Go1BridgeCrossing"` to `_envs` and `_cfgs` in `locomotion/__init__.py`, pointing to `go1_bridge.BridgeCrossing` and `go1_bridge.default_config`.

---

## If Training Goes Badly — Things To Try

- **Observation history stacking**: `history_len=1` exists in `default_config()` but is not implemented anywhere — it's a dead config field. If the robot struggles to detect lateral drift or gait phase, wire up a rolling obs buffer in `reset()` and `step()`. Each step shifts the buffer and inserts the new obs; the policy receives all `history_len` frames flattened (46 × history_len dimensions). The neural network input size must be updated to match. Start with `history_len=5` (5 previous timesteps).

---

## What Is Still TBD

- **`fall_threshold`**: 0.2 is a starting guess. Sweep `{0.1, 0.2, 0.35}` in early training runs and pick the value that gives clean episode boundaries without spurious truncations.

- **Reward scales**: `forward_vel`, `orientation`, `alive` scales are initial guesses. Inspect per-component reward magnitudes early in training and retune.

- **No domain randomisation**: A `randomize.py` is not needed for initial training. If sim-to-real transfer becomes a goal later, one can be added following the go1 joystick pattern.
