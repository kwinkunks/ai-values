#!/usr/bin/env python3
"""
Paper figure: zero-shot vs conversation questionnaires, for the models run BOTH ways
(GPT-3.5, GPT-4, Mistral S 3).
  Left  — map positions: each model's zero-shot and conversation mean with its 80%
          ellipse, joined by a connector. Positions are similar; the conversation
          ellipse is tighter.
  Right — coherence: per-model spread across personas (persona-SD). Conversation is
          consistently lower — more self-consistent responses.

Usage: uv run --with matplotlib --with scipy python figures/plot_zeroshot.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

PAIRED = {'gpt-3.5-turbo': 'GPT-3.5', 'gpt-4-turbo': 'GPT-4', 'mistral-small-2506': 'Mistral S 3'}
ZS_COL, CONV_COL = '#d62728', '#2a78d6'   # zero-shot red, conversation blue


def main():
    V = C.VARIABLES
    exp = json.load(open(C.EXPERIMENTS_FILE))
    m, s, w = np.loadtxt(C.MEANS_FILE), np.loadtxt(C.SDS_FILE), np.loadtxt(C.WEIGHTS_FILE)
    df = pd.read_csv(C.RESPONSES_FILE).dropna(subset=V)
    Z = ((df[V].values - m) / s) @ w
    df = df.assign(x=1.81 * Z[:, 0] + 0.038, y=1.61 * Z[:, 1] - 0.1)
    df['model'] = df['experiment_id'].astype(str).map({k: v['model'] for k, v in exp.items()})
    df['zs'] = df['experiment_id'].astype(str).map({k: bool(v.get('zero_shot')) for k, v in exp.items()}).astype(bool)
    df['persona'] = df['system'].map(C._persona)

    rows = {}
    for mod in PAIRED:
        g = df[df['model'] == mod]
        rows[mod] = {}
        for mode, sub in (('zs', g[g['zs']]), ('conv', g[~g['zs']])):
            pm = sub.groupby('persona')[['x', 'y']].mean()
            rows[mod][mode] = dict(mean=pm.mean().to_numpy(),
                                   sd=float(np.sqrt(pm['x'].var(ddof=1) + pm['y'].var(ddof=1))),
                                   ell=C.hotelling_ellipse(pm['x'].values, pm['y'].values, 0.80))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 7.2), dpi=200,
                                   gridspec_kw={'width_ratios': [1.35, 1]})
    fig.patch.set_facecolor(base.SURFACE)

    # -- Panel A: map --
    for mod, name in PAIRED.items():
        for mode, col in (('zs', ZS_COL), ('conv', CONV_COL)):
            r = rows[mod][mode]; mu = r['mean']; e = r['ell']
            if e:
                axA.add_patch(Ellipse(mu, 2 * e['ea'], 2 * e['eb'], angle=np.degrees(e['etheta']),
                                      facecolor=col, alpha=0.12, edgecolor=col, lw=1.4, zorder=5))
            axA.scatter(*mu, s=90, c=col, edgecolors='white', linewidths=1.2, zorder=7)
        z, c = rows[mod]['zs']['mean'], rows[mod]['conv']['mean']
        axA.annotate('', xy=c, xytext=z, arrowprops=dict(arrowstyle='-|>', color=base.MUTED,
                                                         lw=1.4, alpha=0.8, shrinkA=7, shrinkB=7), zorder=6)
        axA.annotate(name, c, fontsize=9, fontweight='bold', color=base.INK, zorder=8,
                     xytext=(7, 4), textcoords='offset points',
                     path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
    base.draw_base_map(axA, grey=True, label_countries=['United States', 'China'],
                       extra_handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=ZS_COL,
                                             markeredgecolor='none', markersize=9, label='zero-shot'),
                                      Line2D([0], [0], marker='o', color='w', markerfacecolor=CONV_COL,
                                             markeredgecolor='none', markersize=9, label='conversation')])
    axA.set_title('Position: zero-shot vs conversation (80% ellipses)', fontsize=12.5, color=base.INK)

    # -- Panel B: coherence (persona spread) --
    axB.set_facecolor(base.SURFACE)
    names = list(PAIRED.values())
    for i, mod in enumerate(PAIRED):
        zsd, csd = rows[mod]['zs']['sd'], rows[mod]['conv']['sd']
        axB.plot([zsd, csd], [i, i], color=base.MUTED, lw=2, zorder=2)
        axB.scatter([zsd], [i], s=110, c=ZS_COL, zorder=3, edgecolors='white', linewidths=1)
        axB.scatter([csd], [i], s=110, c=CONV_COL, zorder=3, edgecolors='white', linewidths=1)
        axB.annotate(f'{csd/zsd:.0%} of zero-shot', ((zsd + csd) / 2, i + 0.16), fontsize=8.5,
                     color=base.INK, ha='center')
    axB.set_yticks(range(len(names))); axB.set_yticklabels(names, fontsize=11, color=base.INK)
    axB.set_ylim(-0.5, len(names) - 0.3)
    axB.set_xlim(0, max(rows[m]['zs']['sd'] for m in PAIRED) * 1.15)
    axB.set_xlabel('Spread across personas  (persona-SD, map units)', fontsize=11, color=base.INK)
    axB.set_title('Coherence: conversation gives tighter responses', fontsize=12.5, color=base.INK)
    axB.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=ZS_COL, markersize=10, label='zero-shot'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=CONV_COL, markersize=10, label='conversation')],
               loc='lower right', fontsize=9, frameon=True, facecolor=base.SURFACE, edgecolor=base.GRID)
    axB.grid(axis='x', color=base.GRID, lw=0.5, alpha=0.6, zorder=0)
    axB.tick_params(colors=base.MUTED)
    for sp in axB.spines.values():
        sp.set_color(base.GRID)

    fig.suptitle('Zero-shot vs conversation questionnaires  (the 3 models run both ways)',
                 fontsize=13.5, color=base.INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / 'zeroshot_vs_conversation.png'
    fig.savefig(out, dpi=200, facecolor=base.SURFACE, bbox_inches='tight')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
