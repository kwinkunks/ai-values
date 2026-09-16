#!/usr/bin/env python3
"""
Paper figures: one model lineage (family) on the cultural map, in the GPT-figure style
(grey countries, black Norway, US/China labelled; each model coloured with all its
records and its 80% confidence ellipse). Produces two figures per lineage:
  llms_<slug>_first.png — the earliest release only, labelled with its release month.
  llms_<slug>_all.png   — every model, labelled with release month, joined by a
                          release-ordered trajectory (arrow to newest); colour = time.

Usage:
  uv run --with matplotlib python figures/plot_llms_lineage.py --lineage "Claude Opus"
  uv run --with matplotlib python figures/plot_llms_lineage.py --lineage "Claude Sonnet"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

FIG_DIR = Path(__file__).parent
sys.path.insert(0, str(FIG_DIR))
sys.path.insert(0, str(FIG_DIR.parent / 'src'))
import plot_cultural_map as base  # noqa: E402
import compare as C  # noqa: E402

TRAJ_COLOR = '#444444'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lineage', default='Claude Opus',
                   help="family name: used for titles/slug, and (unless --models) to filter by config `lineage`")
    p.add_argument('--models', nargs='*', default=None,
                   help="explicit list of model labels (overrides the lineage filter; for GPT etc.)")
    p.add_argument('--label-left', nargs='*', default=[],
                   help="model labels whose on-plot text goes to the LEFT of the point")
    p.add_argument('--label-dy', type=float, default=0.0,
                   help="vertical nudge for ALL on-plot labels, in map units (negative = down)")
    p.add_argument('--label-dy-for', nargs='*', default=[],
                   help="per-model vertical nudges as LABEL=DY (map units, negative = down); adds to --label-dy")
    p.add_argument('--legend', action=argparse.BooleanOptionalAction, default=True,
                   help="draw the legend (default on; use --no-legend to hide)")
    p.add_argument('--trajectory', action=argparse.BooleanOptionalAction, default=True,
                   help="draw the release-ordered trajectory line (default on)")
    p.add_argument('--first', action=argparse.BooleanOptionalAction, default=True,
                   help="also render the earliest-model-only figure (default on)")
    p.add_argument('--out', default=None, help="filename for the combined figure (default llms_<slug>_all.png)")
    p.add_argument('--title', default=None, help="title override for the combined figure")
    p.add_argument('--color', nargs='*', default=[],
                   help="per-model colour overrides as LABEL=HEX (e.g. 'GPT-5=#d62728')")
    args = p.parse_args()
    left_set = set(args.label_left)
    dy_for = {k: float(v) for k, v in (pair.split('=', 1) for pair in args.label_dy_for)}
    slug = args.lineage.split()[-1].lower()

    exp = json.load(open(C.EXPERIMENTS_FILE))
    rel = {v['label']: v.get('release', '') for v in exp.values()}
    lin = {v['label']: v.get('lineage', '') for v in exp.values()}
    w, m, s = (np.loadtxt(C.WEIGHTS_FILE), np.loadtxt(C.MEANS_FILE), np.loadtxt(C.SDS_FILE))
    llm = C.compute_llm_coordinates(C.load_responses(latest_only=True), exp, w, m, s).set_index('country')

    pool = args.models if args.models else [lab for lab in llm.index if lin.get(lab) == args.lineage]
    models = sorted([lab for lab in pool if lab in llm.index and rel.get(lab)], key=lambda lab: rel[lab])
    if not models:
        sys.exit(f'No models found for {args.lineage!r} with release dates.')
    if args.models:
        missing = [lab for lab in args.models if lab not in models]
        if missing:
            print(f'  (not found / no release, skipped: {missing})')
    # viridis over the non-overridden models (keeps their shades stable), then apply
    # explicit --color LABEL=HEX overrides (e.g. highlight GPT-5 in red).
    overrides = dict(pair.split('=', 1) for pair in args.color)
    ramp = [lab for lab in models if lab not in overrides]
    color = {lab: plt.cm.viridis(t) for lab, t in zip(ramp, np.linspace(0.12, 0.78, len(ramp)))}
    color.update(overrides)

    def render(subset, out, title, trajectory):
        fig, ax = plt.subplots(figsize=(11, 8.6), dpi=200)
        fig.patch.set_facecolor(base.SURFACE)
        handles, means = [], []
        for lab in subset:
            r = llm.loc[lab]
            col = color[lab]
            pts = np.array(r['pts']) if isinstance(r['pts'], list) else np.empty((0, 2))
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=15, color=col, alpha=0.25, edgecolors='none', zorder=4)
            if r['ea'] == r['ea']:
                ax.add_patch(Ellipse((r['x'], r['y']), 2 * r['ea'], 2 * r['eb'],
                                     angle=np.degrees(r['etheta']), facecolor=col, alpha=0.12,
                                     edgecolor=col, lw=1.5, zorder=5))
            means.append((r['x'], r['y']))
            short = lab.replace('Claude ', '')
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=col,
                                  markeredgecolor='none', markersize=9, label=f'{short}  ({rel[lab][:7]})'))

        if trajectory and len(means) > 1:
            xs, ys = zip(*means)
            ax.plot(xs, ys, color=TRAJ_COLOR, lw=1.8, alpha=0.8, zorder=6)
            ax.annotate('', xy=means[-1], xytext=means[-2],
                        arrowprops=dict(arrowstyle='-|>', color=TRAJ_COLOR, lw=1.8), zorder=6)

        for lab in subset:
            r = llm.loc[lab]
            ax.scatter([r['x']], [r['y']], s=95, color=color[lab], edgecolors='white',
                       linewidths=1.2, zorder=7)
            left = lab in left_set
            ax.annotate(f"{lab.replace('Claude ', '')}  {rel[lab][:7]}",
                        (r['x'], r['y'] + args.label_dy + dy_for.get(lab, 0.0)),
                        fontsize=8.5, fontweight='bold', color=color[lab], zorder=8,
                        xytext=(-7 if left else 7, 4), textcoords='offset points',
                        ha='right' if left else 'left',
                        path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])

        base.draw_base_map(ax, extra_handles=handles, grey=True,
                           label_countries=['United States', 'China'], legend=args.legend)
        ax.set_title(title, fontsize=14, color=base.INK, pad=12)
        fig.tight_layout()
        fig.savefig(FIG_DIR / out, dpi=200, facecolor=base.SURFACE, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {FIG_DIR / out}')

    if args.first:
        first = models[0]
        render([first], f'llms_{slug}_first.png',
               f'{first} on the cultural map (all records + 80% ellipse)', trajectory=False)
    render(models, args.out or f'llms_{slug}_all.png',
           args.title or f'{args.lineage} over time (all records, 80% ellipses, release trajectory)',
           trajectory=args.trajectory)


if __name__ == '__main__':
    main()
