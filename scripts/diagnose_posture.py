"""Measure trunk posture (pitch, roll, height) and reward breakdown of a policy.

Runs a batch of push-free episodes with the deterministic policy and reports,
separately for steps on platform A (x < 0) and on the beam (0.5 < x < 3.5):
  pitch_deg   trunk nose-down angle (+ = nose down)
  roll_deg    trunk roll
  height_m    trunk z above the 0.5 m walking surface
  joint_dev   mean joint angle minus the home keyframe, per joint (rad)
  com_x_rel   whole-robot CoM x minus mean foot x (m, + = CoM ahead of feet)
  rewards     mean per-step value of every reward term (already scaled)

Example:
  python scripts/diagnose_posture.py --env_name Go1BridgeCrossingTail \
    --checkpoint logs/<run>/checkpoints/<step> \
    --overrides '{"tail_mode": "active", "bridge_half_width": 0.05, "virtual_hw_min": 0.04}' \
    --out results/posture/active_A07.json
"""

import argparse
import json
import pathlib

import jax
import jax.numpy as jp
import numpy as np
from brax.training import checkpoint as brax_checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

JOINTS = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
          "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--env_name", required=True)
  p.add_argument("--checkpoint", default="",
                 help="Empty = zero action (hold the home pose).")
  p.add_argument("--overrides", default="{}")
  p.add_argument("--num_envs", type=int, default=128)
  p.add_argument("--episode_length", type=int, default=1500)
  p.add_argument("--out", required=True)
  args = p.parse_args()

  ovr = json.loads(args.overrides)
  ovr.update({"push_config.enable": True, "push_config.interval": [1e6, 1e6],
              "episode_length": args.episode_length})
  env = registry.load(args.env_name, config_overrides=ovr)
  if args.checkpoint:
    params = brax_checkpoint.load(pathlib.Path(args.checkpoint).resolve())
    nets = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **dict(locomotion_params.brax_ppo_config(args.env_name).network_factory))
    policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
  else:
    policy = lambda obs, key: (  # noqa: E731
        jp.zeros((obs["state"].shape[0], env.action_size)), None)

  torso = env._torso_body_id  # pylint: disable=protected-access
  feet = env._feet_site_id  # pylint: disable=protected-access
  default = np.asarray(env._default_pose)[:12]  # pylint: disable=protected-access

  def features(state):
    d = state.data
    r = d.xmat[torso]
    fwd, left = r[:, 0], r[:, 1]
    pitch = jp.rad2deg(-jp.arcsin(jp.clip(fwd[2], -1, 1)))
    roll = jp.rad2deg(jp.arcsin(jp.clip(left[2], -1, 1)))
    com_x = d.subtree_com[torso, 0]
    feet_x = jp.mean(d.site_xpos[feet, 0])
    rw = {k[len("reward/"):]: v for k, v in state.metrics.items()
          if k.startswith("reward/")}
    out = {"pitch_deg": pitch, "roll_deg": roll, "height_m": d.qpos[2] - 0.5,
           "com_x_rel": com_x - feet_x, "x": d.qpos[0],
           "joints": d.qpos[7:19] - default}
    if d.qpos.shape[0] > 19:
      out["tail_deg"] = jp.rad2deg(d.qpos[19])
    return out, rw

  reset = jax.jit(jax.vmap(env.reset))
  step = jax.jit(jax.vmap(env.step))
  infer = jax.jit(policy)
  feat = jax.jit(jax.vmap(features))

  key = jax.random.PRNGKey(0)
  state = reset(jax.random.split(key, args.num_envs))
  done = np.zeros(args.num_envs, bool)
  rows = []
  for k in range(args.episode_length):
    key, sub = jax.random.split(key)
    act, _ = infer(state.obs, sub)
    state = step(state, act)
    f, rw = feat(state)
    f = jax.tree.map(np.asarray, f)
    rw = jax.tree.map(np.asarray, rw)
    live = ~done
    rows.append((live.copy(), f, rw))
    done |= np.asarray(state.done) > 0
    if done.all():
      break

  def summarise(mask_fn):
    acc, rew, n = {}, {}, 0
    for live, f, rw in rows:
      m = live & mask_fn(f["x"])
      if not m.any():
        continue
      n += int(m.sum())
      for name, v in f.items():
        acc.setdefault(name, []).append(v[m])
      for name, v in rw.items():
        rew.setdefault(name, []).append(v[m])
    if n == 0:
      return {"n_steps": 0}
    cat = {k: np.concatenate(v) for k, v in acc.items()}
    res = {"n_steps": n}
    for k in ("pitch_deg", "roll_deg", "height_m", "com_x_rel", "tail_deg"):
      if k in cat:
        res[k] = {"mean": float(cat[k].mean()), "std": float(cat[k].std()),
                  "p10": float(np.percentile(cat[k], 10)),
                  "p90": float(np.percentile(cat[k], 90))}
    res["joint_dev_rad"] = dict(zip(JOINTS, np.round(cat["joints"].mean(0), 3).tolist()))
    res["rewards"] = {k: float(np.concatenate(v).mean()) for k, v in rew.items()}
    return res

  summary = {
      "env_name": args.env_name, "checkpoint": args.checkpoint,
      "overrides": json.loads(args.overrides),
      "platform": summarise(lambda x: x < -0.2),
      "beam": summarise(lambda x: (x > 0.5) & (x < 3.5)),
  }
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(summary, indent=2))
  for zone in ("platform", "beam"):
    s = summary[zone]
    if s["n_steps"]:
      print(f"{zone}: pitch {s['pitch_deg']['mean']:+.1f} deg, "
            f"height {s['height_m']['mean']:.3f} m, "
            f"com_x_rel {s['com_x_rel']['mean']:+.3f} m")


if __name__ == "__main__":
  main()
