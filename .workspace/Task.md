# Narrow Bridge Locomotion Task

## Overview

A locomotion challenge where a legged robot (humanoid or quadruped) must traverse a narrow bridge connecting two platforms. The bridge width is a tunable parameter that directly controls task difficulty, enabling curriculum learning from easy to hard.

---

## Environment Description

### Terrain Layout

```
[  Platform A  ]----[  bridge  ]----[  Platform B  ]
     (start)       (width = W)           (goal)
```

- **Platform A** — wide starting platform where the robot is spawned
- **Bridge** — narrow elevated walkway of configurable width `W`
- **Platform B** — wide goal platform the robot must reach
- Falling off any edge terminates the episode

### Key Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| Bridge width `W` | 0.1m → 0.8m | Core difficulty dial |
| Bridge length `L` | Fixed (e.g. 4m) | Can be extended for harder tasks |
| Bridge height `H` | Fixed (e.g. 0.5m) | Fall penalty severity |

---

## Robot

- **Quadruped** — Unitree Go1 *(easier baseline, recommended first)*
- **Humanoid** — Unitree H1 *(harder, higher reward ceiling)*
- Loaded from standard MuJoCo MJCF/URDF model files

---

## Task Objective

Control the robot from Platform A to Platform B without falling, as efficiently and quickly as possible.
