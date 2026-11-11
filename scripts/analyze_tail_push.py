"""Summarise the tail vs no-tail push-tolerance experiment.

Reads results/tail_push/<condition>/{manifest.tsv, eval/*.json} and writes:
  results/tail_push/summary.md          tables: J50 per condition x width, stage success
  results/tail_push/survival_curves.png survival rate vs push impulse, one panel per width
  results/tail_push/j50_vs_width.png    critical impulse J50 vs beam width

Usage: python scripts/analyze_tail_push.py [--root results/tail_push]
"""

import argparse
import csv
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (validated: CVD dE >= 9; aqua < 3:1 contrast, so
# every series is also direct-labelled and has its own marker shape).
CONDITIONS = [
    ("active", "Active tail", "#2a78d6", "o"),
    ("locked", "Locked tail", "#eb6834", "s"),
    ("notail", "No tail", "#1baf7a", "^"),
]
WIDTHS = [("w020", 0.20), ("w010", 0.10), ("w006", 0.06)]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"


def _load(root: pathlib.Path):
  evals, manifests = {}, {}
  for cond, *_ in CONDITIONS:
    for f in sorted((root / cond / "eval").glob("*.json")):
      evals[(cond, f.stem)] = json.loads(f.read_text())
    m = root / cond / "manifest.tsv"
    if m.exists():
      with m.open() as fh:
        manifests[cond] = [r for r in csv.reader(fh, delimiter="\t") if r]
  return evals, manifests


def _fmt_j50(v):
  if v is None:
    return "n/a"
  if isinstance(v, float) and math.isinf(v):
    return "> sweep"
  return f"{v:.1f}"


def _style(ax):
  ax.set_facecolor(SURFACE)
  ax.grid(True, color=GRID, linewidth=0.8)
  ax.set_axisbelow(True)
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
  ax.tick_params(colors=MUTED, labelsize=9)


def _survival_figure(evals, out: pathlib.Path):
  fig, axes = plt.subplots(1, len(WIDTHS), figsize=(13, 4.2), sharey=True)
  fig.patch.set_facecolor(SURFACE)
  for ax, (tag, width) in zip(axes, WIDTHS):
    _style(ax)
    for cond, label, color, marker in CONDITIONS:
      e = evals.get((cond, f"B_{tag}"))
      if not e:
        continue
      xs = [r["impulse_Ns"] for r in e["results"]]
      ys = [100 * r["survive_rate"] for r in e["results"]]
      ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=6,
              markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
      # No direct labels here: curves converge at 0%/100% and labels collide.
      # Identity = shared legend + a distinct marker shape per condition.
    ax.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_title(f"{width:.2f} m beam", fontsize=11, color=INK)
    ax.set_xlabel("Push impulse (N·s, 0.1 s pulse)", fontsize=9, color=INK)
  axes[0].set_ylabel("Episodes survived (%)", fontsize=9, color=INK)
  axes[0].set_ylim(-3, 103)
  handles, labels = axes[0].get_legend_handles_labels()
  if handles:
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
  fig.suptitle("Survival after one lateral push, after push training",
               y=1.09, fontsize=12, color=INK)
  fig.tight_layout()
  fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
  plt.close(fig)


def _j50_figure(evals, out: pathlib.Path):
  fig, ax = plt.subplots(figsize=(6.5, 4.2))
  fig.patch.set_facecolor(SURFACE)
  _style(ax)
  for cond, label, color, marker in CONDITIONS:
    pts = []
    for tag, width in WIDTHS:
      e = evals.get((cond, f"B_{tag}"))
      if e and e["critical_impulse_J50_Ns"] is not None:
        j = e["critical_impulse_J50_Ns"]
        if math.isinf(j):
          j = max(r["impulse_Ns"] for r in e["results"])
        pts.append((width, j))
    if not pts:
      continue
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.annotate(label, (xs[0], ys[0]), xytext=(6, 0),
                textcoords="offset points", fontsize=8, color=INK, va="center")
  ax.set_xlabel("Beam width (m)", fontsize=9, color=INK)
  ax.set_ylabel("Critical push J50 (N·s)", fontsize=9, color=INK)
  ax.set_title("Push needed to make the robot fall half the time",
               fontsize=11, color=INK)
  ax.legend(frameon=False, fontsize=9)
  fig.tight_layout()
  fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", default="results/tail_push")
  root = pathlib.Path(ap.parse_args().root)
  evals, manifests = _load(root)

  lines = ["# Tail push experiment summary", "",
           "## Critical push impulse J50 (N·s) — 50% survival, 0.1 s lateral push", "",
           "| Width | " + " | ".join(f"{l} (before → after push training)" for _, l, _, _ in CONDITIONS) + " |",
           "|---|" + "---|" * len(CONDITIONS)]
  a_stage = {"w020": "A04_w020", "w010": "A07_w010", "w006": "A11_w006"}
  for tag, width in WIDTHS:
    cells = []
    for cond, *_ in CONDITIONS:
      a = evals.get((cond, a_stage[tag]))
      b = evals.get((cond, f"B_{tag}"))
      cells.append(
          f"{_fmt_j50(a and a['critical_impulse_J50_Ns'])} → "
          f"{_fmt_j50(b and b['critical_impulse_J50_Ns'])}"
      )
    lines.append(f"| {width:.2f} m | " + " | ".join(cells) + " |")

  lines += ["", "## Survival rate (%) vs impulse, after push training", ""]
  for tag, width in WIDTHS:
    imps = None
    rows = []
    for cond, label, *_ in CONDITIONS:
      e = evals.get((cond, f"B_{tag}"))
      if not e:
        continue
      imps = imps or [r["impulse_Ns"] for r in e["results"]]
      rows.append(f"| {label} | " + " | ".join(
          f"{100 * r['survive_rate']:.0f}" for r in e["results"]) + " |")
    if imps:
      lines += [f"**{width:.2f} m beam**", "",
                "| Condition | " + " | ".join(f"{i:g}" for i in imps) + " |",
                "|---|" + "---|" * len(imps)] + rows + [""]

  lines += ["## Training success rate per stage (last eval)", "",
            "| Stage | " + " | ".join(l for _, l, _, _ in CONDITIONS) + " |",
            "|---|" + "---|" * len(CONDITIONS)]
  stages = []
  for cond in manifests:
    for r in manifests[cond]:
      if r[0] not in stages:
        stages.append(r[0])
  for s in stages:
    cells = []
    for cond, *_ in CONDITIONS:
      rate = next((r[1] for r in manifests.get(cond, []) if r[0] == s), "")
      cells.append(f"{float(rate):.2f}" if rate else "–")
    lines.append(f"| {s} | " + " | ".join(cells) + " |")

  (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  _survival_figure(evals, root / "survival_curves.png")
  _j50_figure(evals, root / "j50_vs_width.png")
  print(f"wrote {root / 'summary.md'}, survival_curves.png, j50_vs_width.png")


if __name__ == "__main__":
  main()
