# Phase: Go1 Environment Module

## Fall / Termination Logic

The bridge task needs two termination conditions (combined with `|`):

1. **Tilt termination** — `upvector[-1] < 0.0` (robot pitched past horizontal). Matches the pattern used in `go1/joystick.py:306`. Works for falls on the bridge surface itself.

2. **Height termination** — `robot_z < fall_threshold` where `robot_z` is the root body z-position. Needed when the robot walks off the edge of the bridge or a platform, since tilt alone won't fire until after the robot has already fallen a significant distance.

### fall_threshold

The value is **TBD — requires empirical testing**. A starting suggestion from the terrain plan is `0.2` (i.e. 0.3m below the platform top surface at z=0.5m), but this may not be effective in practice:

- Too high: triggers termination while the robot is still recoverable (e.g. leaning over the edge).
- Too low: lets the robot simulate a long fall before resetting, wasting training steps.

Test a range (e.g. `0.1`, `0.2`, `0.35`) in early training runs and pick whichever gives the cleanest episode boundaries without spurious truncations. Make `fall_threshold` a named field in `default_config()` so it can be tuned without touching the environment code.
