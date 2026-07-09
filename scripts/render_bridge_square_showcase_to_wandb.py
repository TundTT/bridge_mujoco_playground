#!/usr/bin/env python3
"""Render selected bridge policies with presentational bridge geometry."""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import os
from pathlib import Path
from typing import Any

# MuJoCo must see this before the mujoco module is imported.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from brax.training import checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np
import wandb

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params


SHOWCASE_STAGES = (
    {
        "stage": 1,
        "width_label": "0.4m",
        "artifact": "bridge-policy-stage01-width-0p4m",
    },
    {
        "stage": 4,
        "width_label": "0.2m",
        "artifact": "bridge-policy-stage04-width-0p20m",
    },
    {
        "stage": 7,
        "width_label": "0.1m",
        "artifact": "bridge-policy-stage07-width-0p10m",
    },
    {
        "stage": 10,
        "width_label": "0.07m",
        "artifact": "bridge-policy-stage10-width-0p07m",
    },
    {
        "stage": 30,
        "width_label": "0.01m",
        "artifact": "bridge-policy-stage30-width-0p01m",
    },
    {
        "stage": 33,
        "width_label": "0.0025m",
        "artifact": "bridge-policy-stage33-width-0p0025m",
    },
)


def _width_m(width_label: str) -> float:
  return float(width_label.removesuffix("m"))


def _width_slug(width_label: str) -> str:
  return width_label.replace(".", "p").removesuffix("m")


def _resolve_checkpoint(
    api: wandb.Api,
    entity: str,
    source_project: str,
    artifact_name: str,
    artifact_root: Path,
) -> tuple[Path, dict[str, Any]]:
  artifact = api.artifact(f"{entity}/{source_project}/{artifact_name}:latest")
  metadata = dict(artifact.metadata)
  local_path = Path(metadata.get("local_checkpoint_path", ""))
  if local_path.is_absolute() and local_path.exists():
    return local_path, metadata

  download_dir = artifact_root / artifact_name
  artifact_dir = Path(artifact.download(root=download_dir))
  checkpoint_path = artifact_dir / "checkpoint"
  if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Downloaded artifact {artifact_name} but did not find {checkpoint_path}"
    )
  return checkpoint_path.resolve(), metadata


def _make_policy(checkpoint_path: Path, env) -> Any:
  params = checkpoint.load(checkpoint_path.resolve())
  rl_config = locomotion_params.brax_ppo_config("Go1BridgeCrossing")
  network_factory_kwargs = dict(rl_config.network_factory)
  networks = ppo_networks.make_ppo_networks(
      env.observation_size,
      env.action_size,
      preprocess_observations_fn=running_statistics.normalize,
      **network_factory_kwargs,
  )
  return ppo_networks.make_inference_fn(networks)(params, deterministic=True)


def _light_state(state):
  empty_data = state.data.__class__(
      **{key: None for key in state.data.__annotations__}
  )
  empty_state = state.__class__(
      **{key: None for key in state.__annotations__}
  )
  return empty_state.replace(data=empty_data)


def _rollout_success(env, policy, seed: int, max_steps: int):
  reset = jax.jit(env.reset)
  step = jax.jit(env.step)
  infer = jax.jit(policy)
  state = reset(jax.random.PRNGKey(seed))
  template = _light_state(state)
  trajectory = []
  total_reward = 0.0
  rng = jax.random.PRNGKey(seed + 10_000)

  for _ in range(max_steps):
    rng, action_key = jax.random.split(rng)
    action, _ = infer(state.obs, action_key)
    state = step(state, action)
    total_reward += float(state.reward)
    trajectory.append(
        template.tree_replace({
            "data.qpos": state.data.qpos,
            "data.qvel": state.data.qvel,
            "data.time": state.data.time,
            "data.ctrl": state.data.ctrl,
            "data.mocap_pos": state.data.mocap_pos,
            "data.mocap_quat": state.data.mocap_quat,
            "data.xfrc_applied": state.data.xfrc_applied,
        })
    )
    if bool(state.done):
      return (
          bool(state.metrics["metric/term_success"]),
          trajectory,
          total_reward,
          float(state.metrics["metric/max_x_reached"]),
      )

  return False, trajectory, total_reward, float(state.metrics["metric/max_x_reached"])


def _choose_rollout(env, policy, seed: int, max_steps: int, max_attempts: int):
  best = None
  for attempt in range(max_attempts):
    result = _rollout_success(env, policy, seed + attempt, max_steps)
    success, trajectory, total_reward, max_x = result
    print(
        f"  attempt {attempt + 1}/{max_attempts}: "
        f"success={success} reward={total_reward:.3f} max_x={max_x:.3f} "
        f"steps={len(trajectory)}"
    )
    if best is None or max_x > best[3]:
      best = result
    if success:
      return attempt + 1, result
  assert best is not None
  return max_attempts, best


def _make_bridge_modifier(env, bridge_width_m: float, beam_shape: str):
  bridge_id = env.mj_model.geom("bridge").id
  radius = bridge_width_m / 2.0
  center_z = 0.5 - radius

  def modify_scene(scene: mujoco.MjvScene) -> None:
    for geom_idx in range(scene.ngeom):
      geom = scene.geoms[geom_idx]
      if (
          geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
          and geom.objid == bridge_id
      ):
        if beam_shape == "square":
          geom.type = mujoco.mjtGeom.mjGEOM_BOX
          geom.size[:] = np.array([2.0, radius, radius])
          geom.pos[:] = np.array([2.0, 0.0, center_z])
        elif beam_shape == "round":
          mujoco.mjv_connector(
              geom,
              mujoco.mjtGeom.mjGEOM_CYLINDER,
              radius,
              np.array([0.0, 0.0, center_z]),
              np.array([4.0, 0.0, center_z]),
          )
        else:
          raise ValueError(f"Unsupported beam shape: {beam_shape}")
        geom.rgba[:] = np.array([0.95, 0.64, 0.18, 1.0])
        return

  return modify_scene


def _render_video(
    env,
    trajectory,
    bridge_width_m: float,
    beam_shape: str,
    output_path: Path,
    camera: str,
    height: int,
    width: int,
    render_every: int,
) -> float:
  fps = 1.0 / env.dt / render_every
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  sampled = trajectory[::render_every]
  modifier = _make_bridge_modifier(env, bridge_width_m, beam_shape)
  frames = env.render(
      sampled,
      height=height,
      width=width,
      camera=camera,
      scene_option=scene_option,
      modify_scene_fns=[modifier] * len(sampled),
  )
  media.write_video(str(output_path), frames, fps=fps)
  return fps


def _format_width_label(width_m: float) -> str:
  return f"{width_m:g}m"


def _visual_geometry_description(beam_shape: str) -> str:
  if beam_shape == "square":
    return "box square cross-section, top flush with platforms"
  if beam_shape == "round":
    return "cylinder round cross-section, top tangent flush with platforms"
  raise ValueError(f"Unsupported beam shape: {beam_shape}")


def _log_per_width_runs(
    args: argparse.Namespace,
    manifest: list[dict[str, Any]],
    summary_run_id: str,
) -> None:
  for row in manifest:
    width_label = _format_width_label(row["bridge_width_m"])
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=f"{args.beam_shape}_beam_{width_label}",
        group=f"selected_{args.beam_shape}_beam_showcase",
        job_type="showcase-video",
        tags=["showcase", f"{args.beam_shape}-beam", f"width-{width_label}"],
        config={
            "source_project": args.source_project,
            "source_run_id": row["source_run_id"],
            "source_run_name": row["source_run_name"],
            "stage": row["stage"],
            "bridge_width_m": row["bridge_width_m"],
            "bridge_height_m": row["bridge_height_m"],
            "bridge_diameter_m": row["bridge_width_m"],
            "bridge_length_m": 4.0,
            "beam_shape": args.beam_shape,
            "physics_beam_shape": args.physics_beam_shape,
            "visual_geometry": _visual_geometry_description(args.beam_shape),
            "summary_showcase_run_id": summary_run_id,
        },
        reinit=True,
    )
    video_path = Path(row["video"])
    wandb.log({
        f"{args.beam_shape}_beam_rollout": wandb.Video(
            str(video_path), format="mp4"
        )
    })
    run.summary["success"] = row["success"]
    run.summary["attempts"] = row["attempts"]
    run.summary["episode_reward"] = row["episode_reward"]
    run.summary["max_x_reached"] = row["max_x_reached"]
    run.summary["bridge_width_m"] = row["bridge_width_m"]
    run.summary["bridge_height_m"] = row["bridge_height_m"]
    run.summary["local_video"] = str(video_path)
    print(f"logged per-width run {run.name}: {run.url}")
    wandb.finish()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--entity", default="Tund")
  parser.add_argument("--source-project", default="bridge_crossing_final_v2")
  parser.add_argument("--project", default="bridge_crossing_final_showcase")
  parser.add_argument("--run-name", default=None)
  parser.add_argument("--beam-shape", choices=("square", "round"), default="square")
  parser.add_argument(
      "--physics-beam-shape",
      choices=("box", "round"),
      default="box",
      help=(
          "Actual MJX collision geometry used for rollout. Use round to test "
          "whether a policy generalizes to a physical cylinder beam."
      ),
  )
  parser.add_argument("--camera", default="track")
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--render-every", type=int, default=2)
  parser.add_argument("--max-steps", type=int, default=2000)
  parser.add_argument("--max-attempts", type=int, default=20)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument(
      "--log-per-width-runs",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Also create one finished W&B run per selected bridge width.",
  )
  args = parser.parse_args()

  timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  output_dir = (
      args.output_dir
      or Path("logs") / f"bridge_{args.beam_shape}_showcase-{timestamp}"
  ).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  artifact_root = output_dir / "checkpoint_artifacts"
  api = wandb.Api()

  run = wandb.init(
      entity=args.entity,
      project=args.project,
      name=args.run_name or f"{args.beam_shape}_beam_selected_widths",
      job_type="showcase-render",
      tags=["showcase", f"{args.beam_shape}-beam", "go1", "bridge"],
      config={
          "source_project": args.source_project,
          "camera": args.camera,
          "height": args.height,
          "width": args.width,
          "render_every": args.render_every,
          "max_steps": args.max_steps,
          "beam_shape": args.beam_shape,
          "bridge_visual": _visual_geometry_description(args.beam_shape),
          "physics_beam_shape": args.physics_beam_shape,
          "bridge_top_z_m": 0.5,
      },
  )
  table = wandb.Table(
      columns=[
          "stage",
          "bridge_width_m",
          "bridge_height_m",
          "success",
          "attempts",
          "episode_reward",
          "max_x_reached",
          "source_run_id",
          "checkpoint",
          "video",
      ]
  )
  manifest = []
  summary_run_id = run.id

  try:
    for item in SHOWCASE_STAGES:
      width_m = _width_m(item["width_label"])
      half_width = width_m / 2.0
      checkpoint_path, metadata = _resolve_checkpoint(
          api,
          args.entity,
          args.source_project,
          item["artifact"],
          artifact_root,
      )
      env = registry.load(
          "Go1BridgeCrossing",
          config=registry.get_default_config("Go1BridgeCrossing"),
          config_overrides={
              "impl": "jax",
              "bridge_half_width": half_width,
              "virtual_hw_min": half_width,
              "bridge_shape": args.physics_beam_shape,
          },
      )
      policy = _make_policy(checkpoint_path, env)
      attempts, result = _choose_rollout(
          env,
          policy,
          seed=args.seed + int(item["stage"]) * 100,
          max_steps=args.max_steps,
          max_attempts=args.max_attempts,
      )
      success, trajectory, total_reward, max_x = result
      video_path = (
          output_dir
          / f"bridge_{_width_slug(item['width_label'])}m_{args.beam_shape}.mp4"
      )
      fps = _render_video(
          env,
          trajectory,
          width_m,
          args.beam_shape,
          video_path,
          camera=args.camera,
          height=args.height,
          width=args.width,
          render_every=args.render_every,
      )
      video = wandb.Video(str(video_path), fps=int(fps), format="mp4")
      key = f"bridge_{_width_slug(item['width_label'])}m_{args.beam_shape}"
      wandb.log({key: video}, step=int(item["stage"]))
      table.add_data(
          item["stage"],
          width_m,
          width_m,
          success,
          attempts,
          total_reward,
          max_x,
          metadata.get("source_run_id"),
          str(checkpoint_path),
          wandb.Video(str(video_path), fps=int(fps), format="mp4"),
      )
      manifest.append({
          "stage": item["stage"],
          "bridge_width_m": width_m,
          "bridge_height_m": width_m,
          "bridge_diameter_m": width_m,
          "beam_shape": args.beam_shape,
          "physics_beam_shape": args.physics_beam_shape,
          "success": success,
          "attempts": attempts,
          "episode_reward": total_reward,
          "max_x_reached": max_x,
          "source_run_id": metadata.get("source_run_id"),
          "source_run_name": metadata.get("source_run_name"),
          "checkpoint": str(checkpoint_path),
          "video": str(video_path),
      })
      print(
          f"rendered {item['width_label']}: success={success} "
          f"attempts={attempts} reward={total_reward:.3f} video={video_path}"
      )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    wandb.log({"showcase_videos": table})
    wandb.save(str(manifest_path), base_path=str(output_dir))
    run.summary["num_videos"] = len(manifest)
    run.summary["all_successful"] = all(row["success"] for row in manifest)
    run.summary["output_dir"] = str(output_dir)
  finally:
    wandb.finish()

  if args.log_per_width_runs and manifest:
    _log_per_width_runs(args, manifest, summary_run_id)


if __name__ == "__main__":
  main()
