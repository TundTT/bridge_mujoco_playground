"""How much do the legs (thighs, knees, calves) pass through the beam?

Rolls out a policy on its own env config (physics as trained), then replays each
step in a copy of the model with the Go2 leg-collision capsules switched on
and records leg-vs-bridge penetration. Uses the leg capsules only to measure;
they do not act on the rollout unless --leg_collision is given.

Example:
  python scripts/check_leg_penetration.py --env_name Go2BridgeCrossingTail \
    --checkpoint <ckpt> --overrides '{"tail_mode": "active", "bridge_half_width": 0.1}' \
    --num_envs 64 --episode_length 1500 --label active_B_w020
"""

import argparse
import json
import pathlib

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training import checkpoint as brax_checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--env_name", required=True)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--overrides", default="{}")
  p.add_argument("--num_envs", type=int, default=64)
  p.add_argument("--episode_length", type=int, default=1500)
  p.add_argument("--sample_every", type=int, default=5)
  p.add_argument("--leg_collision", action="store_true",
                 help="also turn the leg capsules on in the rollout physics")
  p.add_argument("--label", default="")
  p.add_argument("--out", default=None, help="append a JSON line here")
  a = p.parse_args()

  ovr = json.loads(a.overrides)
  ovr.update({"push_config.enable": True, "push_config.interval": [1e6, 1e6],
              "episode_length": a.episode_length})
  if a.leg_collision:
    ovr["leg_collision"] = True
  env = registry.load(a.env_name, config_overrides=ovr)
  probe = registry.load(a.env_name, config_overrides={**ovr, "leg_collision": True})
  pm = probe.mj_model
  pd = mujoco.MjData(pm)
  bridge = pm.geom("bridge").id
  leg_ids = {i for i in range(pm.ngeom) if pm.geom(i).name.endswith("_legcol")}

  params = brax_checkpoint.load(pathlib.Path(a.checkpoint).resolve())
  cfg = locomotion_params.brax_ppo_config(a.env_name)
  nets = ppo_networks.make_ppo_networks(
      env.observation_size, env.action_size,
      preprocess_observations_fn=running_statistics.normalize,
      **dict(cfg.network_factory))
  infer = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
  reset = jax.jit(jax.vmap(env.reset))
  step = jax.jit(jax.vmap(env.step))

  key = jax.random.PRNGKey(0)
  state = reset(jax.random.split(key, a.num_envs))
  alive = np.ones(a.num_envs, bool)
  n_samples = n_pen = n_pen5 = 0
  worst = 0.0
  per_leg = {}
  for k in range(a.episode_length):
    key, kk = jax.random.split(key)
    act, _ = infer(state.obs, kk)
    state = step(state, act)
    alive &= np.asarray(state.done) == 0
    if k % a.sample_every or not alive.any():
      continue
    qpos = np.asarray(state.data.qpos)
    for e in np.flatnonzero(alive):
      if not 0.0 <= qpos[e, 0] <= 4.0:  # only while the trunk is over the beam
        continue
      pd.qpos[:] = qpos[e]
      mujoco.mj_forward(pm, pd)
      n_samples += 1
      depth = 0.0
      for c in pd.contact[:pd.ncon]:
        g = {c.geom1, c.geom2}
        if bridge in g and g & leg_ids:
          leg = pm.geom(next(iter(g & leg_ids))).name
          depth = max(depth, -c.dist)
          per_leg[leg] = max(per_leg.get(leg, 0.0), -c.dist)
      n_pen += depth > 0.002
      n_pen5 += depth > 0.01
      worst = max(worst, depth)
    if not alive.any():
      break
  row = {"label": a.label, "env_name": a.env_name, "checkpoint": a.checkpoint,
         "leg_collision_in_rollout": a.leg_collision,
         "samples_over_beam": n_samples,
         "frac_penetrating_gt2mm": n_pen / max(n_samples, 1),
         "frac_penetrating_gt10mm": n_pen5 / max(n_samples, 1),
         "max_depth_m": worst,
         "max_depth_by_geom_m": dict(sorted(per_leg.items(), key=lambda t: -t[1])[:4])}
  print(json.dumps(row), flush=True)
  if a.out:
    with open(a.out, "a") as f:
      f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
  main()
