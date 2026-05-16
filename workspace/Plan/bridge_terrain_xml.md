# Phase: Bridge Terrain XML

## Goal

Create `scene_mjx_feetonly_bridge.xml` in the Go1 xmls directory. This scene composes the existing Go1 robot with a three-part bridge terrain (Platform A → narrow bridge beam → Platform B). The bridge geom is named so Python code can resize its width for curriculum learning without reloading the model.

---

## File to create

```
mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_feetonly_bridge.xml
```

---

## Terrain layout

```
[  Platform A  ]----[    bridge    ]----[  Platform B  ]
     (start)        (width = W, 4m)          (goal)
      x=-3..0            x=0..4               x=4..7
                    z = 0.5m top surface
```

Robot spawns on Platform A facing +x. Goal is Platform B. Falling off any edge ends the episode (detected in Python by height check, no floor geom needed).

---

## Geometry

Two MuJoCo-specific things to know before reading the table:

**Static geometry** — In MuJoCo, a `<geom>` inside a `<body>` moves with that body. A `<geom>` placed directly inside `<worldbody>` (the root) is permanently fixed in the world — it never moves regardless of physics. Platforms and bridge should never move, so they go directly in `<worldbody>` with no parent body and no joints.

**Half-extents** — MuJoCo box `size` is not width/height/depth — it's the distance from the center to each face. So `size="1.5 1.0 0.25"` means the box extends 1.5m in ±x, 1.0m in ±y, and 0.25m in ±z, giving a full footprint of 3m × 2m × 0.5m. For the bridge geom, the y half-extent is the curriculum dial — the Python env shrinks it each stage to make the bridge narrower (e.g. `y=0.2` → 0.4m wide, `y=0.05` → 0.1m wide). Make sure this value is exposed as a named config parameter (`bridge_width`) so the curriculum can adjust it without digging into the XML or geometry math.

| Geom name    | `pos`   | `size`      | Footprint |
|-----------------|---------------|----------------|---------------|
| `platform_a` | `-1.5 0 0.25` | `1.5 1.0 0.25` | 3×2×0.5 m   |
| `bridge`     | `2.0 0 0.25`  | `2.0 0.2 0.25` | 4×0.4×0.5 m |
| `platform_b` | `5.5 0 0.25`  | `1.5 1.0 0.25` | 3×2×0.5 m   |

The `bridge` geom name is the contract with Python — `mj_model.geom("bridge").id` is how Python looks up the geom to patch its size.

Default bridge half-width `0.2` (= 0.4m full width) is a medium-difficulty starting point. The Python env will override this at model-load time via `config.bridge_width`.

---

## Collision model

Follow the existing "feetonly" convention used in all other Go1 scenes:

- **Terrain geoms**: `contype="1" conaffinity="0"` — floor-like, collidable
- **Feet**: `contype="0" conaffinity="1"` — already set in `go1_mjx_feetonly.xml`
- **All other body geoms**: `contype="0" conaffinity="0"` — already set in `go1_mjx_feetonly.xml`

This means only feet-to-terrain contacts are simulated; body self-collision and body-terrain collision are disabled. This keeps the contact count low and simulation fast.

A `<default class="terrain">` block sets `contype`/`conaffinity` once, all three platform/bridge geoms inherit it.

---

## Keyframe

The `home` keyframe spawns the robot on Platform A:

- `x = -1.5` (center of platform_a)
- `y = 0`
- `z = 0.278 + 0.5 = 0.778` (robot standing height above platform surface)
- Quaternion `1 0 0 0` (no rotation, facing +x)
- Joint angles and ctrl copied from the existing flat terrain `home` keyframe

```xml
<key name="home"
  qpos="-1.5 0 0.778  1 0 0 0  0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"
  ctrl="0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"/>
```

---

## Camera

The `statistic` block controls MuJoCo viewer framing. Center it on the bridge midpoint:

```xml
<statistic center="2 0 0.5" extent="5" meansize="0.04"/>
```

The robot XML already defines `track`, `top`, `side`, and `back` cameras attached to the trunk — those are inherited and work without changes.

---

## Visual / materials

Two named materials to distinguish terrain visually:

- `platform` — light grey `rgba="0.75 0.75 0.75 1"`
- `bridge` — warm tan `rgba="0.85 0.65 0.35 1"` (easy to spot in viewer)

---

## Full XML structure

```xml
<mujoco model="go1 bridge scene">
  <include file="go1_mjx_feetonly.xml"/>

  <statistic center="2 0 0.5" extent="5" meansize="0.04"/>

  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <rgba force="1 0 0 1"/>
    <global azimuth="120" elevation="-20"/>
    <map force="0.01"/>
    <scale forcewidth="0.3" contactwidth="0.5" contactheight="0.2"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <material name="platform" rgba="0.75 0.75 0.75 1" reflectance="0.3"/>
    <material name="bridge"   rgba="0.85 0.65 0.35 1" reflectance="0.3"/>
  </asset>

  <default>
    <default class="terrain">
      <geom type="box" contype="1" conaffinity="0"/>
    </default>
  </default>

  <worldbody>
    <geom name="platform_a" class="terrain" material="platform"
          pos="-1.5 0 0.25" size="1.5 1.0 0.25"/>
    <geom name="bridge"     class="terrain" material="bridge"
          pos="2.0  0 0.25" size="2.0 0.2 0.25"/>
    <geom name="platform_b" class="terrain" material="platform"
          pos="5.5  0 0.25" size="1.5 1.0 0.25"/>
  </worldbody>

  <include file="sensor_feet.xml"/>

  <keyframe>
    <key name="home"
      qpos="-1.5 0 0.778  1 0 0 0  0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"
      ctrl="0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"/>
  </keyframe>
</mujoco>
```

---

## Open questions

1. **Robot spawn z-height**: `0.778` assumes Go1 standing height 0.278m. Should verify against the `home` keyframe in `scene_mjx_feetonly_flat_terrain.xml` (which uses `z=0.27`). May need a small upward nudge to avoid spawn-in-contact jitter.

2. **Platform dimensions**: 3m long × 2m wide is generous for a Go1 (~0.3m wide). Could shrink platform width for a harder task or extend for easier spawning. Keep it generous for now.

3. **Bridge-to-platform seam**: Box geoms placed edge-to-edge (`platform_a` ends at x=0, `bridge` starts at x=0). No gap or overlap by construction. Verify in viewer that there's no z-fighting or step artifact at the join.

4. **No floor plane**: Omitting the infinite `plane` geom means there's nothing to catch the robot after a fall. That's intentional — fall detection is done in Python (`done = robot_z < fall_threshold`). Confirm the fall threshold value (suggestion: `z < 0.2`, i.e. 0.3m below platform surface).

5. **Curriculum width patching**: The Python env's `__init__` should patch `geom_size[bridge_id]` from `config.bridge_width` immediately after `mjx.put_model(mj_model)`. This is not in scope for this XML phase but should be noted as the next step.
