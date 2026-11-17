"""Video of the standing push test: same push, active tail vs locked tail vs no tail.

Each panel is one episode of the stand task on the box beam. At --push_time the
trunk gets one 0.1 s half-sine lateral push of --impulse N*s (the same push
eval_push_tolerance.py applies). The red arrow shows the push force. The
camera looks along the beam from behind the robot, so roll and the tail swing are both
visible.

Example (on the workstation; mediapy needs ffmpeg on PATH):
  MUJOCO_GL=egl python scripts/render_stand_push.py \
    --panel active:Go2BridgeCrossingTail:<ckpt> \
    --panel locked:Go2BridgeCrossingTail:<ckpt> \
    --panel notail:Go2BridgeCrossing:<ckpt> \
    --hw 0.1 --vhw_min 0.05 --impulse 15 --side right --out stand_push.mp4
"""

import argparse
import json
import pathlib

import jax
import jax.numpy as jp
import mediapy
import mujoco
import numpy as np
from brax.training import checkpoint as brax_checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from PIL import Image, ImageDraw, ImageFont

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

_LABELS = {"active": "Active tail", "locked": "Locked tail", "notail": "No tail"}
_COND_OVR = {
    "active": {"tail_mode": "active"},
    "locked": {"tail_mode": "locked"},
    "notail": {"push_config.enable": True, "reset_info_on_new_episode": True},
}
_STAND_OVR = {
    "task": "stand",
    "reward_config.scales.alive": 10.0,
    "reward_config.scales.stand_pos": -10.0,
    "reward_config.scales.frontier_delta": 0.0,
    "reward_config.scales.success": 0.0,
    "reward_config.scales.feet_air_time": 0.0,
}


def _parse_args():
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--panel", action="append", required=True,
                 help="condition:env_name:checkpoint (repeat per panel)")
  p.add_argument("--hw", type=float, default=0.1, help="bridge_half_width (m)")
  p.add_argument("--vhw_min", type=float, default=0.05)
  p.add_argument("--impulse", type=float, default=15.0, help="N*s")
  p.add_argument("--duration", type=float, default=0.1, help="push length (s)")
  p.add_argument("--side", choices=("left", "right"), default="right")
  p.add_argument("--leg_collision", action="store_true",
                 help="turn on Go2 leg-vs-beam collision (match how the policy was trained)")
  p.add_argument("--push_time", type=float, default=2.0, help="s")
  p.add_argument("--start", type=float, default=1.5, help="video start (s)")
  p.add_argument("--end", type=float, default=6.0, help="video end (s)")
  p.add_argument("--slowmo", type=float, default=2.0, help="playback slow-down")
  p.add_argument("--seeds", type=int, default=8,
                 help="try this many seeds; use the first where the first "
                      "panel survives (all panels share that seed)")
  p.add_argument("--width", type=int, default=480)
  p.add_argument("--height", type=int, default=400)
  p.add_argument("--out", required=True)
  return p.parse_args()


def _policy(env, env_name, ckpt):
  params = brax_checkpoint.load(pathlib.Path(ckpt).resolve())
  cfg = locomotion_params.brax_ppo_config(env_name)
  nets = ppo_networks.make_ppo_networks(
      env.observation_size, env.action_size,
      preprocess_observations_fn=running_statistics.normalize,
      **dict(cfg.network_factory))
  return jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))


def _pulse_gain(duration, dt):
  t_mid = (np.arange(int(np.ceil(duration / dt)) + 1) + 0.5) * dt
  t_mid = t_mid[t_mid < duration]
  return float(np.sum(np.sin(np.pi * t_mid / duration)) * dt)


def _rollout(cond, env_name, ckpt, args, seed):
  ovr = {**_COND_OVR[cond], **_STAND_OVR,
         "bridge_half_width": args.hw, "virtual_hw_min": args.vhw_min,
         "push_config.enable": True, "push_config.interval": [1e6, 1e6],
         "push_config.impulse": [0.0, 0.0]}
  if args.leg_collision:
    ovr["leg_collision"] = True
  env = registry.load(env_name, config_overrides=ovr)
  infer = _policy(env, env_name, ckpt)
  reset, step = jax.jit(env.reset), jax.jit(env.step)
  dt = env.dt
  sign = 1.0 if args.side == "left" else -1.0
  peak = args.impulse / _pulse_gain(args.duration, dt) * np.array([0.0, sign, 0.0])
  n_steps = int(round(args.end / dt))
  push_step = int(round(args.push_time / dt))
  key = jax.random.PRNGKey(seed)
  state = reset(key)
  traj, fell_at = [], None
  for k in range(n_steps):
    if k == push_step:
      info = dict(state.info)
      info["push_t"] = jp.array(0.0)
      info["push_dur"] = jp.array(args.duration)
      info["push_peak"] = jp.array(peak, dtype=jp.float32)
      info["push_wait"] = jp.array(1e6)
      state = state.replace(info=info)
    key, k_act = jax.random.split(key)
    action, _ = infer(state.obs, k_act)
    state = step(state, action)
    force = np.asarray(state.data.xfrc_applied[env._torso_body_id, :3])  # pylint: disable=protected-access
    traj.append((np.asarray(state.data.qpos), np.asarray(state.data.qvel), force))
    if fell_at is None and float(state.done) > 0:
      fell_at = k  # keep simulating so the fall is visible
  return env, traj, fell_at


def _frame(renderer, model, data, qpos, force, lookat, args):
  data.qpos[:] = qpos
  mujoco.mj_forward(model, data)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = lookat
  cam.azimuth, cam.elevation, cam.distance = 0.0, -15.0, 1.35  # from behind
  renderer.update_scene(data, cam)
  f = np.linalg.norm(force)
  if f > 1.0:  # push arrow, pointing along the force, ending at the trunk
    scn = renderer.scene
    if scn.ngeom < scn.maxgeom:
      base = data.xpos[model.body("base").id]
      length = 0.12 + 0.35 * f / 300.0
      tip = base + np.array([0.0, 0.0, 0.03]) - 0.12 * force / f
      tail = tip - length * force / f
      g = scn.geoms[scn.ngeom]
      mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                          np.zeros(3), np.zeros(9),
                          np.array([0.9, 0.1, 0.1, 1.0], dtype=np.float32))
      mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.025, tail, tip)
      scn.ngeom += 1
  return renderer.render()


def _label(img, lines):
  im = Image.fromarray(img)
  d = ImageDraw.Draw(im)
  try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    small = ImageFont.truetype("DejaVuSans.ttf", 16)
  except OSError:
    font = small = ImageFont.load_default()
  y = 8
  for i, (text, color) in enumerate(lines):
    d.text((10, y), text, fill=color, font=font if i == 0 else small,
           stroke_width=2, stroke_fill=(255, 255, 255))
    y += 26 if i == 0 else 21
  return np.asarray(im)


def main():
  args = _parse_args()
  panels = [p.split(":", 2) for p in args.panel]

  # Choose a seed where the first panel (the active tail) survives the push.
  chosen = None
  for seed in range(args.seeds):
    env, traj, fell = _rollout(*panels[0], args, seed)
    print(f"seed {seed}: {panels[0][0]} fell_at={fell}", flush=True)
    if fell is None:
      chosen = (seed, env, traj, fell)
      break
  if chosen is None:
    chosen = (0, *_rollout(*panels[0], args, 0))
  seed = chosen[0]
  runs = [chosen[1:]] + [_rollout(*p, args, seed) for p in panels[1:]]

  dt = runs[0][0].dt
  i0, i1 = int(round(args.start / dt)), int(round(args.end / dt))
  push_k = int(round(args.push_time / dt))
  frames = []
  renderers, datas = [], []
  for env, _, _ in runs:
    renderers.append(mujoco.Renderer(env.mj_model, args.height, args.width))
    datas.append(mujoco.MjData(env.mj_model))
  summary = {"seed": seed, "impulse_Ns": args.impulse, "side": args.side,
             "hw": args.hw, "panels": []}
  for (cond, _, _), (env, traj, fell) in zip(panels, runs):
    q = np.stack([t[0] for t in traj])
    has_tail = env.mj_model.nq > 19
    summary["panels"].append({
        "condition": cond, "fell_at_s": None if fell is None else fell * dt,
        "tail_deg_min": float(np.rad2deg(q[:, 19].min())) if has_tail else None,
        "tail_deg_max": float(np.rad2deg(q[:, 19].max())) if has_tail else None,
    })
  print(json.dumps(summary, indent=1), flush=True)

  for k in range(i0, i1):
    row = []
    for (cond, _, _), (env, traj, fell), r, d in zip(panels, runs, renderers, datas):
      kk = min(k, len(traj) - 1)
      qpos, _, force = traj[kk]
      lookat = np.array([traj[0][0][0], 0.0, 0.62])
      img = _frame(r, env.mj_model, d, qpos, force, lookat, args)
      t = (k + 1) * dt
      lines = [(_LABELS.get(cond, cond), (20, 20, 20)),
               (f"t = {t:4.2f} s   ({args.slowmo:g}x slow motion)", (60, 60, 60))]
      if env.mj_model.nq > 19:
        lines.append((f"tail {np.rad2deg(qpos[19]):+5.0f} deg", (20, 60, 160)))
      if push_k <= k < push_k + int(round(args.duration / dt)) + 1:
        arrow = "-->" if args.side == "right" else "<--"
        lines.append((f"PUSH {args.impulse:g} N*s {arrow} (to robot's {args.side})",
                      (200, 20, 20)))
      if fell is not None and k >= fell:
        lines.append(("FELL", (200, 20, 20)))
      row.append(_label(img, lines))
    frames.append(np.concatenate(row, axis=1))
  for r in renderers:
    r.close()

  fps = round(1.0 / dt / args.slowmo)
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  mediapy.write_video(out, frames, fps=fps)
  print(f"wrote {out} ({len(frames)} frames at {fps} fps)")


if __name__ == "__main__":
  main()
