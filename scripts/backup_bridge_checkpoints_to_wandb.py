#!/usr/bin/env python3
"""Upload final bridge policy checkpoints to W&B artifacts.

The training scripts save Orbax checkpoints only on local disk. This helper
finds the completed stage logs, matches each stage to its W&B run, and uploads
the latest checkpoint directory as a restorable W&B artifact.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import wandb


LOGDIR_RE = re.compile(r"Logs are being stored in:\s*(?P<path>.+)")
RUN_URL_RE = re.compile(r"wandb\.ai/[^/]+/[^/]+/runs/(?P<run_id>[A-Za-z0-9_-]+)")
STAGE_LOG_RE = re.compile(r"stage(?P<stage>\d+)\.log$")
RUN_NAME_RE = re.compile(r"final_stage(?P<stage>\d+)_(?P<width>[0-9.]+m)")


def _git_commit(repo_root: Path) -> str | None:
  try:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return None


def _width_slug(width: str) -> str:
  return width.replace(".", "p")


def _latest_checkpoint(checkpoints_dir: Path) -> Path | None:
  if not checkpoints_dir.is_dir():
    return None
  checkpoints = [
      path for path in checkpoints_dir.iterdir()
      if path.is_dir() and path.name.isdigit()
  ]
  if not checkpoints:
    return None
  return max(checkpoints, key=lambda path: int(path.name))


def _file_manifest(root: Path) -> list[dict[str, Any]]:
  files = []
  for path in sorted(root.rglob("*")):
    if path.is_file():
      files.append({
          "path": path.relative_to(root).as_posix(),
          "size_bytes": path.stat().st_size,
      })
  return files


def _parse_stage_logs(log_globs: list[str]) -> dict[int, dict[str, Any]]:
  stage_logs: dict[int, dict[str, Any]] = {}
  for pattern in log_globs:
    for log_path_str in sorted(glob.glob(pattern)):
      log_path = Path(log_path_str)
      match = STAGE_LOG_RE.search(log_path.name)
      if not match:
        continue

      stage = int(match.group("stage"))
      logdir = None
      run_id = None
      for line in log_path.read_text(errors="replace").splitlines():
        logdir_match = LOGDIR_RE.search(line)
        if logdir_match:
          logdir = Path(logdir_match.group("path").strip())

        run_match = RUN_URL_RE.search(line)
        if run_match:
          run_id = run_match.group("run_id")

      if logdir and run_id:
        current = stage_logs.get(stage)
        if current is None or log_path.stat().st_mtime > current["log_path"].stat().st_mtime:
          stage_logs[stage] = {
              "stage": stage,
              "stage_log": str(log_path),
              "log_path": log_path,
              "logdir": logdir,
              "run_id": run_id,
          }
  return stage_logs


def _artifact_exists(api: wandb.Api, entity: str, project: str, name: str) -> bool:
  try:
    api.artifact(f"{entity}/{project}/{name}:latest")
    return True
  except wandb.errors.CommError:
    return False


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--entity", default="Tund")
  parser.add_argument("--project", default="bridge_crossing_final_v2")
  parser.add_argument("--repo-root", type=Path, default=Path.cwd())
  parser.add_argument("--min-stage", type=int, default=0)
  parser.add_argument("--max-stage", type=int)
  parser.add_argument(
      "--stage-log-glob",
      action="append",
      default=["/tmp/bridge_final_stage*.log", "/tmp/bridge_extra_narrow_stage*.log"],
      help="Glob for tee logs containing local logdirs and W&B run URLs.",
  )
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--force", action="store_true")
  args = parser.parse_args()

  repo_root = args.repo_root.resolve()
  api = wandb.Api()
  git_commit = _git_commit(repo_root)
  hostname = os.uname().nodename

  stage_logs = _parse_stage_logs(args.stage_log_glob)
  selected_stages = sorted(
      stage for stage in stage_logs
      if stage >= args.min_stage and (args.max_stage is None or stage <= args.max_stage)
  )
  if not selected_stages:
    raise SystemExit("No matching stage logs found.")

  backup_run = None
  try:
    for stage in selected_stages:
      info = stage_logs[stage]
      run = api.run(f"{args.entity}/{args.project}/{info['run_id']}")
      run_name_match = RUN_NAME_RE.search(run.name)
      if not run_name_match:
        print(f"SKIP stage {stage:02d}: cannot parse width from run name {run.name!r}")
        continue

      if run.state != "finished":
        print(f"SKIP stage {stage:02d}: run {run.id} is {run.state}")
        continue

      latest_ckpt = _latest_checkpoint(info["logdir"] / "checkpoints")
      if latest_ckpt is None:
        print(f"SKIP stage {stage:02d}: no checkpoint under {info['logdir'] / 'checkpoints'}")
        continue

      width = run_name_match.group("width")
      artifact_name = f"bridge-policy-stage{stage:02d}-width-{_width_slug(width)}"
      if not args.force and _artifact_exists(api, args.entity, args.project, artifact_name):
        print(f"SKIP stage {stage:02d}: artifact {artifact_name}:latest already exists")
        continue

      checkpoint_step = int(latest_ckpt.name)
      width_m = float(width.removesuffix("m"))
      metadata = {
          "artifact_schema": "bridge_policy_checkpoint_v1",
          "source_entity": args.entity,
          "source_project": args.project,
          "source_run_id": run.id,
          "source_run_name": run.name,
          "source_run_url": run.url,
          "stage": stage,
          "bridge_width_m": width_m,
          "bridge_half_width_m": width_m / 2.0,
          "checkpoint_step": checkpoint_step,
          "local_checkpoint_path": str(latest_ckpt),
          "local_logdir": str(info["logdir"]),
          "stage_log": info["stage_log"],
          "git_commit": git_commit,
          "hostname": hostname,
          "success": run.summary.get("eval/episode_metric/term_success"),
          "episode_reward": run.summary.get("eval/episode_reward"),
          "summary_step": run.summary.get("_step"),
      }
      aliases = [
          "latest",
          f"stage{stage:02d}",
          f"width-{_width_slug(width)}",
          f"run-{run.id}",
      ]

      print(
          "UPLOAD"
          f" stage {stage:02d}"
          f" width={width}"
          f" run={run.id}"
          f" checkpoint={latest_ckpt}"
          f" success={metadata['success']}"
      )
      if args.dry_run:
        continue

      if backup_run is None:
        backup_run = wandb.init(
            entity=args.entity,
            project=args.project,
            job_type="checkpoint-backup",
            name="backup_bridge_policy_checkpoints",
            config={
                "source_project": args.project,
                "min_stage": args.min_stage,
                "max_stage": args.max_stage,
                "git_commit": git_commit,
                "hostname": hostname,
            },
        )

      artifact = wandb.Artifact(
          name=artifact_name,
          type="policy-checkpoint",
          metadata=metadata,
          description=(
              f"Final Orbax PPO policy checkpoint for {run.name} "
              f"from {args.entity}/{args.project}."
          ),
      )
      artifact.add_dir(str(latest_ckpt), name="checkpoint")

      manifest = {
          "metadata": metadata,
          "files": _file_manifest(latest_ckpt),
      }
      with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(manifest, tmp, indent=2, sort_keys=True)
        tmp_path = Path(tmp.name)
      try:
        artifact.add_file(str(tmp_path), name="manifest.json")
        assert backup_run is not None
        backup_run.log_artifact(artifact, aliases=aliases)
      finally:
        tmp_path.unlink(missing_ok=True)
  finally:
    if backup_run is not None:
      backup_run.finish()


if __name__ == "__main__":
  main()
