#!/usr/bin/env python3
"""
Paper figure: zero-shot vs conversation questionnaires, for every model run BOTH ways.
Pairs are matched by LABEL (zero-shot labels are "<name> (zero-shot)", conversation is
"<name>"), which is robust to the two modes using different API model strings.
  Left  — map positions: each model's zero-shot and conversation mean with its 80%
          ellipse, joined by a connector.
  Right — coherence: spread across personas (SD of persona means) per model, zero-shot
          vs conversation, ordered by the ratio.

Reads the raw responses (NOT the zero-shot-filtered loader). Writes
figures/zeroshot_vs_conversation.png.

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

ZS_COL, CONV_COL = '#d62728', '#2a78d6'   # zero-shot red, conversation blue
SUFFIX = ' (zero-shot)'


def main():
    V = C.VARIABLES
    exp = json.load(open(C.EXPERIMENTS_FILE))
    m, s, w = np.loadtxt(C.MEANS_FILE), np.loadtxt(C.SDS_FILE), np.loadtxt(C.WEIGHTS_FILE)
    df = pd.read_csv(C.RESPONSES_FILE).dropna(subset=V)
    Z = ((df[V].values - m) / s) @ w
    df = df.assign(x=1.81 * Z[:, 0] + 0.038, y=1.61 * Z[:, 1] - 0.1)
    df['label'] = df['experiment_id'].astype(str).map({k: v['label'] for k, v in exp.items()})
    df['zs'] = df['experiment_id'].astype(str).map({k: bool(v.get('zero_shot')) for k, v in exp.items()}).astype(bool)
    df['prompt'] = df['system'].map(C._persona)
    df['base'] = df['label'].str.replace(SUFFIX, '', regex=False)

    zbases = set(df[df['zs']]['base']); cbases = set(df[~df['zs']]['base'])
    paired = sorted(zbases & cbases)

    def stats(g):
        pm = g.groupby('prompt')[['x', 'y']].mean()
        return dict(mean=pm.mean().to_numpy(),
                    sd=float(np.sqrt(pm['x'].var(ddof=1) + pm['y'].var(ddof=1))),
                    ell=C.hotelling_ellipse(pm['x'].values, pm['y'].values, 0.80))
    rows = {b: {'zs': stats(df[df['zs'] & (df['base'] == b)]),
                'conv': stats(df[~df['zs'] & (df['base'] == b)])} for b in paired}
    paired.sort(key=lambda b: rows[b]['conv']['sd'] / rows[b]['zs']['sd'])   # tightening → loosening

    print(f"{len(paired)} paired models:")
    for b in paired:
        r = rows[b]; ratio = r['conv']['sd'] / r['zs']['sd']
        print(f"  {b:18} shift={np.hypot(*(r['conv']['mean']-r['zs']['mean'])):.2f}  "
              f"zs-SD={r['zs']['sd']:.2f} conv-SD={r['conv']['sd']:.2f}  conv/zs={ratio:.0%}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 7.4), dpi=200,
                                   gridspec_kw={'width_ratios': [1.35, 1]})
    fig.patch.set_facecolor(base.SURFACE)

    # -- Panel A: map --
    for b in paired:
        for mode, col in (('zs', ZS_COL), ('conv', CONV_COL)):
            r = rows[b][mode]; mu = r['mean']; e = r['ell']
            if e:
                axA.add_patch(Ellipse(mu, 2 * e['ea'], 2 * e['eb'], angle=np.degrees(e['etheta']),
                                      facecolor=col, alpha=0.10, edgecolor=col, lw=1.2, zorder=5))
            axA.scatter(*mu, s=70, c=col, edgecolors='white', linewidths=1.1, zorder=7)
        z, c = rows[b]['zs']['mean'], rows[b]['conv']['mean']
        axA.annotate('', xy=c, xytext=z, arrowprops=dict(arrowstyle='-|>', color=base.MUTED,
                                                         lw=1.2, alpha=0.8, shrinkA=6, shrinkB=6), zorder=6)
        axA.annotate(b, c, fontsize=8, fontweight='bold', color=base.INK, zorder=8,
                     xytext=(6, 3), textcoords='offset points',
                     path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    base.draw_base_map(axA, grey=True, label_countries=['United States', 'China'],
                       extra_handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=ZS_COL,
                                             markeredgecolor='none', markersize=9, label='zero-shot'),
                                      Line2D([0], [0], marker='o', color='w', markerfacecolor=CONV_COL,
                                             markeredgecolor='none', markersize=9, label='conversation')])
    axA.set_title('Position: zero-shot vs conversation (80% ellipses)', fontsize=12.5, color=base.INK)

    # -- Panel B: coherence dumbbell --
    axB.set_facecolor(base.SURFACE)
    for i, b in enumerate(paired):
        zsd, csd = rows[b]['zs']['sd'], rows[b]['conv']['sd']
        axB.plot([zsd, csd], [i, i], color=base.MUTED, lw=2, zorder=2)
        axB.scatter([zsd], [i], s=105, c=ZS_COL, zorder=3, edgecolors='white', linewidths=1)
        axB.scatter([csd], [i], s=105, c=CONV_COL, zorder=3, edgecolors='white', linewidths=1)
        axB.annotate(f'{csd/zsd:.0%}', (max(zsd, csd), i), fontsize=8.5, color=base.INK,
                     va='center', ha='left', xytext=(6, 0), textcoords='offset points')
    axB.axvline(0, color=base.GRID, lw=0)
    axB.set_yticks(range(len(paired))); axB.set_yticklabels(paired, fontsize=10, color=base.INK)
    axB.set_ylim(-0.5, len(paired) - 0.3)
    axB.set_xlim(0, max(max(rows[b]['zs']['sd'], rows[b]['conv']['sd']) for b in paired) * 1.2)
    axB.set_xlabel('Spread across prompts  (SD of prompt means, map units)', fontsize=11, color=base.INK)
    axB.set_title('Prompt spread: conversation tightens older models, loosens frontier ones', fontsize=12, color=base.INK)
    axB.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=ZS_COL, markersize=10, label='zero-shot'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=CONV_COL, markersize=10, label='conversation')],
               loc='lower right', fontsize=9, frameon=True, facecolor=base.SURFACE, edgecolor=base.GRID)
    axB.grid(axis='x', color=base.GRID, lw=0.5, alpha=0.6, zorder=0)
    axB.tick_params(colors=base.MUTED)
    for sp in axB.spines.values():
        sp.set_color(base.GRID)

    fig.suptitle(f'Zero-shot vs conversation questionnaires  ({len(paired)} models run both ways)',
                 fontsize=13.5, color=base.INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / f'zeroshot_vs_conversation.{base.FIG_EXT}'
    fig.savefig(out, dpi=base.DPI, facecolor=base.SURFACE, bbox_inches='tight')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
