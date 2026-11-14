"""Push-tolerance evaluation for the bridge-crossing policies.

For each push impulse in a sweep, runs a batch of episodes on the bridge.
Random training pushes are switched off. When the trunk reaches `trigger_x`
(on the beam), exactly one lateral half-sine push of the given impulse and
duration is applied to the trunk; half the trials push left and half right.

Per impulse it records:
  survive  - did not fall for the rest of the episode
  success  - reached goal_x (crossed the bridge)
  fall_2s  - fell within 2 s of the push (push-induced fall)
and estimates the critical impulse J50: the impulse where survive rate
crosses 50% (linear interpolation).

Example:
  python scripts/eval_push_tolerance.py --env_name Go1BridgeCrossingTail \
    --checkpoint logs/<run>/checkpoints/<step> \
    --overrides '{"bridge_half_width": 0.05, "virtual_hw_min": 0.04}' \
    --impulses 0,2,4,6,8,10,12,14,16,18,20,24 --num_trials 256 \
    --out results/tail_active_hw0.05.json
"""

import argparse
import json
import pathlib
import time

import jax
import jax.numpy as jp
import numpy as np
from brax.training import checkpoint as brax_checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params


def _parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--env_name", required=True)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--overrides", default="{}",
                 help="JSON config overrides; use the training stage's ones.")
  p.add_argument("--impulses", default="0,2,4,6,8,10,12,14,16,18,20,24",
                 help="Comma-separated push impulses in N*s.")
  p.add_argument("--push_duration", type=float, default=0.1, help="s")
  p.add_argument("--trigger_x", type=float, default=1.5,
                 help="Trunk x (m) at which the push is applied. Beam: 0..4.")
  p.add_argument("--num_trials", type=int, default=256,
                 help="Episodes per impulse (even: half left, half right).")
  p.add_argument("--episode_length", type=int, default=1500)
  p.add_argument("--fall_window", type=float, default=2.0, help="s")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", required=True)
  return p.parse_args()


def _load_policy(checkpoint_path: str, env, env_name: str):
  params = brax_checkpoint.load(pathlib.Path(checkpoint_path).resolve())
  rl_config = locomotion_params.brax_ppo_config(env_name)
  networks = ppo_networks.make_ppo_networks(
      env.observation_size,
      env.action_size,
      preprocess_observations_fn=running_statistics.normalize,
      **dict(rl_config.network_factory),
  )
  return ppo_networks.make_inference_fn(networks)(params, deterministic=True)


def _discrete_pulse_gain(duration: float, dt: float) -> float:
  """Sum of the half-sine samples the env applies, times dt (s)."""
  t_mid = (np.arange(int(np.ceil(duration / dt)) + 1) + 0.5) * dt
  t_mid = t_mid[t_mid < duration]
  return float(np.sum(np.sin(np.pi * t_mid / duration)) * dt)


def main() -> None:
  args = _parse_args()
  overrides = json.loads(args.overrides)
  # Keep pushes enabled (same observation sizes as training) but never fire
  # a random one: this script triggers the push itself.
  overrides.update({
      "push_config.enable": True,
      "push_config.interval": [1e6, 1e6],
      "episode_length": args.episode_length,
  })
  env = registry.load(args.env_name, config_overrides=overrides)
  cfg = env._config  # pylint: disable=protected-access
  policy = _load_policy(args.checkpoint, env, args.env_name)

  dt = env.dt
  pulse_gain = _discrete_pulse_gain(args.push_duration, dt)
  fall_window_steps = int(round(args.fall_window / dt))
  n = args.num_trials
  side = jp.where(jp.arange(n) % 2 == 0, 1.0, -1.0)  # +y (left) / -y (right)

  reset = jax.jit(jax.vmap(env.reset))
  step = jax.jit(jax.vmap(env.step))
  infer = jax.jit(policy)

  @jax.jit
  def trigger(state, pushed, peak_force):
    x = state.data.qpos[:, 0]
    fire = (~pushed) & (x >= args.trigger_x) & (state.done == 0)
    peak = peak_force * jp.stack(
        [jp.zeros(n), side, jp.zeros(n)], axis=-1
    )
    info = dict(state.info)
    info["push_t"] = jp.where(fire, 0.0, info["push_t"])
    info["push_dur"] = jp.where(fire, args.push_duration, info["push_dur"])
    info["push_peak"] = jp.where(fire[:, None], peak, info["push_peak"])
    info["push_wait"] = jp.where(fire, 1e6, info["push_wait"])
    return state.replace(info=info), pushed | fire, fire

  impulses = [float(v) for v in args.impulses.split(",")]
  results = []
  key = jax.random.PRNGKey(args.seed)
  for impulse in impulses:
    t0 = time.time()
    key, k_reset, k_act = jax.random.split(key, 3)
    state = reset(jax.random.split(k_reset, n))
    peak_force = impulse / pulse_gain
    pushed = jp.zeros(n, dtype=bool)
    push_step = jp.full(n, -1)
    finished = jp.zeros(n, dtype=bool)
    fell = jp.zeros(n, dtype=bool)
    succeeded = jp.zeros(n, dtype=bool)
    fall_step = jp.full(n, -1)
    max_tail = jp.zeros(n)
    tail_over = jp.zeros(n)
    delivered = jp.zeros(n)  # integral of |push force| dt (self-check)
    has_tail = "metric/tail_angle_abs" in state.metrics

    for k in range(args.episode_length):
      state, pushed, fired = trigger(state, pushed, peak_force)
      push_step = jp.where(fired, k, push_step)
      k_act, k_now = jax.random.split(k_act)
      action, _ = infer(state.obs, k_now)
      state = step(state, action)
      delivered = delivered + (~finished) * state.metrics["metric/push_force"] * dt
      done_now = (state.done > 0) & ~finished
      fell = fell | (done_now & (state.metrics["metric/term_fall"] > 0))
      succeeded = succeeded | (
          done_now & (state.metrics["metric/term_success"] > 0)
      )
      fall_step = jp.where(
          done_now & (state.metrics["metric/term_fall"] > 0), k, fall_step
      )
      if has_tail:
        live = ~finished
        max_tail = jp.where(
            live,
            jp.maximum(max_tail, state.metrics["metric/tail_angle_abs"]),
            max_tail,
        )
        tail_over = tail_over + live * state.metrics["metric/tail_over_limit"]
      finished = finished | (state.done > 0)
      if k % 100 == 99 and bool(finished.all()):
        break

    pushed_np = np.asarray(pushed)
    fell_np = np.asarray(fell)
    valid = pushed_np  # trials that reached the trigger point and got pushed
    fell_soon = (
        fell_np
        & (np.asarray(fall_step) >= 0)
        & (np.asarray(fall_step) - np.asarray(push_step) <= fall_window_steps)
    )
    nv = int(valid.sum())
    row = {
        "impulse_Ns": impulse,
        "peak_force_N": peak_force,
        "delta_v_mps": impulse / env._robot_mass,  # pylint: disable=protected-access
        "n_trials": n,
        "n_pushed": nv,
        "n_fell_before_push": int((fell_np & ~pushed_np).sum()),
        "survive_rate": float((~fell_np & valid).sum() / max(nv, 1)),
        "success_rate": float((np.asarray(succeeded) & valid).sum() / max(nv, 1)),
        "fall_within_window_rate": float((fell_soon & valid).sum() / max(nv, 1)),
        "delivered_impulse_Ns_mean": float(np.asarray(delivered)[valid].mean()) if nv else 0.0,
    }
    for name, mask in (("left", np.asarray(side) > 0), ("right", np.asarray(side) < 0)):
      vm = valid & mask
      row[f"survive_rate_{name}"] = float((~fell_np & vm).sum() / max(vm.sum(), 1))
    if has_tail:
      row["tail_max_angle_deg_mean"] = float(np.asarray(max_tail)[valid].mean()) if nv else 0.0
      row["tail_over_limit_steps_mean"] = float(np.asarray(tail_over)[valid].mean()) if nv else 0.0
    row["seconds"] = round(time.time() - t0, 1)
    results.append(row)
    print(json.dumps(row), flush=True)

  # Critical impulse: first crossing of survive_rate below 0.5.
  # J50 stays None if most episodes ended before reaching the push point
  # (that measures "cannot walk here", not push tolerance).
  j50 = None
  valid = bool(results) and results[0]["n_pushed"] >= 0.5 * results[0]["n_trials"]
  if valid and results[0]["survive_rate"] < 0.5:
    j50 = 0.0  # fails even with the weakest push in the sweep
  elif valid:
    for a, b in zip(results, results[1:]):
      if a["survive_rate"] >= 0.5 > b["survive_rate"]:
        fa, fb = a["survive_rate"], b["survive_rate"]
        j50 = a["impulse_Ns"] + (fa - 0.5) / (fa - fb) * (
            b["impulse_Ns"] - a["impulse_Ns"])
        break
    else:
      j50 = float("inf")  # never dropped below 50% within the sweep

  summary = {
      "env_name": args.env_name,
      "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
      "overrides": json.loads(args.overrides),
      "tail_mode": cfg.get("tail_mode", "none"),
      "bridge_half_width": cfg.bridge_half_width,
      "robot_mass_kg": env._robot_mass,  # pylint: disable=protected-access
      "push_duration_s": args.push_duration,
      "trigger_x": args.trigger_x,
      "critical_impulse_J50_Ns": j50,
      "results": results,
  }
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(summary, indent=2))
  print(f"J50 = {j50} N*s  -> {out}")


if __name__ == "__main__":
  main()
