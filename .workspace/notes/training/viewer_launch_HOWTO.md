# Launching the MuJoCo Viewer

## Quick launch

```sh
cd /Users/Tund/Desktop/bridge_mujoco_playground
.venv/bin/mjpython tmp/view_bridge.py
```

You must use `mjpython` (not plain `python`) — on macOS, MuJoCo's GUI requires it to keep the Cocoa main thread free. The script is at `tmp/view_bridge.py`.

---

## What the script does

- Loads `scene_mjx_feetonly_bridge.xml` via the library's asset loader (handles menagerie paths automatically)
- Resets to the `home` keyframe (robot standing on Platform A)
- Opens a passive viewer and runs the physics loop at ~500 Hz