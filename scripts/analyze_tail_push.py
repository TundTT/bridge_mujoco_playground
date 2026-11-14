"""Summarise the tail vs no-tail push-tolerance experiment.

Reads results/tail_push/<condition>/{manifest.tsv, eval/*.json, eval_fine/*.json}
(fine-sweep JSONs replace coarse ones of the same name) and writes:
  results/tail_push/summary.md                     J50 tables (95% CI), survival, stage success
  results/tail_push/survival_curves.png            survival vs push impulse, one panel per width
  results/tail_push/j50_vs_width.png               J50 vs width, width-branch policies
  results/tail_push/j50_vs_width_generalist.png    J50 vs width, 0.06 m-trained policies

J50 = impulse where survival first drops below 50% (linear interpolation).
The CI comes from re-computing J50 on the Wilson 95% bounds of each point.

Usage: python scripts/analyze_tail_push.py [--root results/tail_push]
       python scripts/analyze_tail_push.py --root results/tail_push_go2
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
# every series also has a legend entry and its own marker shape).
CONDITIONS = [
    ("active", "Active tail", "#2a78d6", "o"),
    ("locked", "Locked tail", "#eb6834", "s"),
    ("notail", "No tail", "#1baf7a", "^"),
]
WIDTHS = [("w020", 0.20), ("w010", 0.10), ("w006", 0.06)]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"
Z95 = 1.96


def _load(root: pathlib.Path):
  evals, manifests, fine = {}, {}, set()
  for cond, *_ in CONDITIONS:
    for sub in ("eval", "eval_fine"):
      for f in sorted((root / cond / sub).glob("*.json")):
        evals[(cond, f.stem)] = json.loads(f.read_text())
        if sub == "eval_fine":
          fine.add((cond, f.stem))
    m = root / cond / "manifest.tsv"
    if m.exists():
      with m.open() as fh:
        manifests[cond] = [r for r in csv.reader(fh, delimiter="\t") if r]
  return evals, manifests, fine


def _wilson(p, n):
  if n == 0:
    return 0.0, 1.0
  d = 1 + Z95**2 / n
  c = (p + Z95**2 / (2 * n)) / d
  h = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / d
  return max(0.0, c - h), min(1.0, c + h)


def _j50_of(xs, ps):
  if ps[0] < 0.5:
    return 0.0
  for (xa, pa), (xb, pb) in zip(zip(xs, ps), zip(xs[1:], ps[1:])):
    if pa >= 0.5 > pb:
      return xa + (pa - 0.5) / (pa - pb) * (xb - xa)
  return float("inf")


def _valid(e):
  """False if most trials fell before the push (J50 would be meaningless)."""
  r0 = e["results"][0]
  return r0["n_pushed"] >= 0.5 * r0["n_trials"]


def j50_ci(e):
  """(J50, lo, hi) in N*s, or None if the eval is invalid."""
  if not e or not _valid(e):
    return None
  rs = e["results"]
  xs = [r["impulse_Ns"] for r in rs]
  ps = [r["survive_rate"] for r in rs]
  bounds = [_wilson(r["survive_rate"], r["n_pushed"]) for r in rs]
  return (_j50_of(xs, ps), _j50_of(xs, [b[0] for b in bounds]),
          _j50_of(xs, [b[1] for b in bounds]))


def _fmt(v):
  if v is None:
    return "fell before push"
  j, lo, hi = v
  if math.isinf(j):
    return "> sweep"
  return f"{j:.1f} [{lo:.1f}, {hi:.1f}]"


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
  xmax = 0.0
  for ax, (tag, width) in zip(axes, WIDTHS):
    _style(ax)
    for cond, label, color, marker in CONDITIONS:
      e = evals.get((cond, f"B_{tag}"))
      if not e or not _valid(e):
        continue
      rs = e["results"]
      xs = [r["impulse_Ns"] for r in rs]
      ys = [100 * r["survive_rate"] for r in rs]
      lo, hi = zip(*[_wilson(r["survive_rate"], r["n_pushed"]) for r in rs])
      ax.fill_between(xs, [100 * v for v in lo], [100 * v for v in hi],
                      color=color, alpha=0.15, linewidth=0)
      ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=5,
              markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
      # No direct labels: curves converge at 0%/100% and labels collide.
      zero = next((x for x, y in zip(xs, ys) if y == 0), xs[-1])
      xmax = max(xmax, zero)
    ax.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_title(f"{width:.2f} m beam", fontsize=11, color=INK)
    ax.set_xlabel("Push impulse (N·s, 0.1 s pulse)", fontsize=9, color=INK)
  for ax in axes:
    ax.set_xlim(-0.3, xmax + 1)
  axes[0].set_ylabel("Episodes survived (%)", fontsize=9, color=INK)
  axes[0].set_ylim(-3, 103)
  handles, labels = axes[0].get_legend_handles_labels()
  if handles:
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
  fig.suptitle("Survival after one lateral push, after push training "
               "(shaded: 95% CI)", y=1.09, fontsize=12, color=INK)
  fig.tight_layout()
  fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
  plt.close(fig)


def _generalist_key(tag):
  """Eval name for the push-trained 0.06 m policy tested on width `tag`."""
  return "B_w006" if tag == "w006" else f"X_B_w006_p20_at_{tag}"


def _j50_figure(evals, out: pathlib.Path, key=lambda tag: f"B_{tag}",
                title="Push needed to make the robot fall half the time"):
  fig, ax = plt.subplots(figsize=(6.5, 4.2))
  fig.patch.set_facecolor(SURFACE)
  _style(ax)
  for cond, label, color, marker in CONDITIONS:
    pts = []
    for tag, width in WIDTHS:
      v = j50_ci(evals.get((cond, key(tag))))
      if v and not math.isinf(v[0]):
        pts.append((width, *v))
    if not pts:
      continue
    xs, ys, lo, hi = zip(*pts)
    ax.errorbar(xs, ys, yerr=[[y - a for y, a in zip(ys, lo)],
                              [b - y for y, b in zip(ys, hi)]],
                color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, capsize=3,
                label=label)
    ax.annotate(label, (xs[0], ys[0]), xytext=(8, 0),
                textcoords="offset points", fontsize=8, color=INK, va="center")
  ax.set_xlabel("Beam width (m)", fontsize=9, color=INK)
  ax.set_ylabel("Critical push J50 (N·s, 95% CI)", fontsize=9, color=INK)
  ax.set_ylim(bottom=0)
  ax.set_title(title, fontsize=11, color=INK)
  ax.legend(frameon=False, fontsize=9)
  fig.tight_layout()
  fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
  plt.close(fig)


def _j50_table(evals, before_key, after_key):
  head = ("| Width | " + " | ".join(
      f"{lab} before → after push training" for _, lab, _, _ in CONDITIONS)
          + " |")
  out = [head, "|---|" + "---|" * len(CONDITIONS)]
  for tag, width in WIDTHS:
    def cell(e):
      return _fmt(j50_ci(e)) if e else "–"
    cells = [f"{cell(evals.get((c, before_key(tag))))} → "
             f"{cell(evals.get((c, after_key(tag))))}"
             for c, *_ in CONDITIONS]
    out.append(f"| {width:.2f} m | " + " | ".join(cells) + " |")
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", default="results/tail_push")
  root = pathlib.Path(ap.parse_args().root)
  evals, manifests, fine = _load(root)
  a_stage = {"w020": "A04_w020", "w010": "A07_w010", "w006": "A11_w006"}

  lines = ["# Tail push experiment summary", "",
           f"Evals from the fine sweep: {len(fine)} of {len(evals)}.", "",
           "## Critical push impulse J50 (N·s) — 50% survival, 0.1 s lateral push",
           "", "Width-branch policies (trained at that width). Brackets: 95% CI.",
           ""]
  lines += _j50_table(evals, lambda t: a_stage[t], lambda t: f"B_{t}")
  lines += ["", "## J50 (N·s) of the 0.06 m-trained policies on every width", "",
            "Same policy on every width, so a stall in one width's curriculum",
            "stage cannot show up as push intolerance.", ""]
  lines += _j50_table(
      evals, lambda t: "A11_w006" if t == "w006" else f"X_A11_at_{t}",
      _generalist_key)

  lines += ["", "## Survival rate (%) vs impulse, after push training", ""]
  for tag, width in WIDTHS:
    imps, rows = None, []
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
            "| Stage | " + " | ".join(lab for _, lab, _, _ in CONDITIONS) + " |",
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
  _j50_figure(evals, root / "j50_vs_width_generalist.png", key=_generalist_key,
              title="Critical push, 0.06 m-trained policy on each width")
  print(f"wrote {root / 'summary.md'}, survival_curves.png, j50_vs_width*.png")


if __name__ == "__main__":
  main()
