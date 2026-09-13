#!/usr/bin/env python3
"""
Paper figure: which survey questions separate the chatbots?
For each of the 10 items, every model's mean answer is standardised (SD from the human
population average, using the same means/sds as the projection) so the 1-10 / 1-4 / -2..2
scales are comparable. One row per question, ordered by between-model spread (most
discriminating at the top); each dot is a model, coloured by vendor. The vertical line at
0 is the human average. Writes figures/question_separation.png.

Usage: uv run --with matplotlib python figures/plot_question_separation.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG_DIR = Path(__file__).parent
sys.path.insert(0, str(FIG_DIR))
sys.path.insert(0, str(FIG_DIR.parent / 'src'))
import plot_cultural_map as base  # noqa: E402
import compare as C  # noqa: E402
import compute_weights as CW  # noqa: E402  (axis assignment + weight signs)

SHORT = {'F063': 'God important', 'Y003': 'Child autonomy', 'F120': 'Abortion OK',
         'G006': 'National pride', 'E018': 'Respect authority', 'Y002': 'Country aims',
         'A008': 'Happiness', 'F118': 'Homosexuality OK', 'E025': 'Signed petition',
         'A165': 'Trust people'}
VENDOR_COLOR = {'OpenAI': '#1f77b4', 'Anthropic': '#d62728', 'Google': '#2ca02c',
                'xAI': '#9467bd', 'Mistral': '#ff7f0e', 'Qwen': '#17becf',
                'Microsoft': '#8c564b'}
OTHER = '#9a9a9a'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--orient', action=argparse.BooleanOptionalAction, default=True,
                   help="flip each item by its axis-weight sign so right = self-expression / secular")
    args = p.parse_args()

    V = C.VARIABLES
    exp = json.load(open(C.EXPERIMENTS_FILE))
    m, s = np.loadtxt(C.MEANS_FILE), np.loadtxt(C.SDS_FILE)
    df = C.load_responses(latest_only=True).dropna(subset=V)
    df['label'] = df['experiment_id'].astype(str).map({k: v['label'] for k, v in exp.items()})
    df = df[df['label'].notna()]
    vendor = {v['label']: v.get('vendor', 'Other') for v in exp.values()}

    mm = df.groupby('label')[V].mean()
    Z = pd.DataFrame((mm.values - m) / s, index=mm.index, columns=V)

    # Each item drives one map axis (col 0 = survival↔self-expression, col 1 = trad↔secular).
    # Orient each item by the sign of its weight on that axis, so + always means toward the
    # top-right of the map (self-expression for x-items, secular for y-items).
    weights = np.loadtxt(C.WEIGHTS_FILE)
    axis = {q: (0 if q in CW.SURV_ITEMS else 1) for q in V}
    sign = {q: (np.sign(weights[i, axis[q]]) or 1.0) for i, q in enumerate(V)}
    if args.orient:
        Z = Z.mul(pd.Series(sign))
    axis_name = {q: ('self-expr' if axis[q] == 0 else 'secular') for q in V}

    order = Z.std(ddof=1).sort_values().index.tolist()   # least→most; plotted bottom→top

    fig, ax = plt.subplots(figsize=(12, 7.6), dpi=200)
    fig.patch.set_facecolor(base.SURFACE); ax.set_facecolor(base.SURFACE)
    ax.axvline(0, color=base.MUTED, lw=1.2, zorder=2)
    ax.annotate('human average', (0.05, len(order) - 0.32), fontsize=8.5, color=base.MUTED,
                ha='left', va='top')

    rng = np.random.default_rng(0)
    vcol = {lab: VENDOR_COLOR.get(vendor.get(lab, 'Other'), OTHER) for lab in Z.index}
    for i, q in enumerate(order):
        ax.axhspan(i - 0.5, i + 0.5, color=base.GRID, alpha=0.18 if i % 2 else 0, zorder=0)
        z = Z[q].values
        jit = (rng.random(len(z)) - 0.5) * 0.62
        ax.scatter(z, i + jit, s=26, c=[vcol[l] for l in Z.index], alpha=0.75,
                   edgecolors='white', linewidths=0.4, zorder=3)
        ax.annotate(f'SD {Z[q].std(ddof=1):.2f}', (ax_max := 2.05, i), fontsize=8.5,
                    color=base.INK, va='center', ha='left')

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f'{SHORT[q]}  ({axis_name[q]})' for q in order], fontsize=10.5, color=base.INK)
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_xlim(-2.1, 2.4)
    xlab = ('Model mean, oriented so right = self-expression / secular  (SD from human average)'
            if args.orient else 'Model mean answer — standardised (SD from human average)')
    ax.set_xlabel(xlab, fontsize=11, color=base.INK)
    ax.set_title('Which questions separate the chatbots?  (rows ordered by between-model spread)',
                 fontsize=13, color=base.INK, pad=12)
    ax.tick_params(colors=base.MUTED)
    for sp in ax.spines.values():
        sp.set_color(base.GRID)
    ax.grid(axis='x', color=base.GRID, lw=0.5, alpha=0.6, zorder=0)

    vendors_present = [v for v in VENDOR_COLOR if any(vendor.get(l) == v for l in Z.index)]
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=VENDOR_COLOR[v],
                      markeredgecolor='none', markersize=8, label=v) for v in vendors_present]
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=OTHER,
                          markeredgecolor='none', markersize=8, label='Other'))
    ax.legend(handles=handles, loc='lower left', fontsize=8.5, frameon=True, facecolor=base.SURFACE,
              edgecolor=base.GRID, ncol=2, title=f'Vendor ({len(Z)} models)', title_fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / 'question_separation.png'
    fig.savefig(out, dpi=200, facecolor=base.SURFACE, bbox_inches='tight')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
