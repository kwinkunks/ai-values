#!/usr/bin/env python3
"""
Which questions drive the English→Norwegian shift? For every model with both an EN
(generic) and a NO (generic) run, decompose its map shift into per-question
contributions: for question i, contribution to Δx = 1.81·w[i,0]·(z_i^NO − z_i^EN) and to
Δy = 1.61·w[i,1]·(z_i^NO − z_i^EN), where z = (mean score − population mean)/sd. These
sum exactly to the model's Δx / Δy. Bars = mean across models; dots = each model (by
vendor), so you see both the driver and how consistent it is. Writes out/no_contrib.png.

Usage: uv run --with matplotlib python src/plot_contrib.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import compare as C  # noqa: E402

SHORT = {  # variable -> short label for the y-axis
    'F063': 'God important', 'Y003': 'Child autonomy', 'F120': 'Abortion OK',
    'G006': 'National pride', 'E018': 'Respect authority', 'Y002': 'Country aims',
    'A008': 'Happiness', 'F118': 'Homosexuality OK', 'E025': 'Signed petition',
    'A165': 'Trust people',
}
VENDOR_COLOR = {'OpenAI': '#fb923c', 'Anthropic': '#c2410c', 'Microsoft': '#2563eb'}
INK, MUTED, GRID, SURFACE, BAR = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb', '#c9c8c2'


def main():
    V = C.VARIABLES
    exp = json.load(open(C.EXPERIMENTS_FILE))
    w = np.loadtxt(C.WEIGHTS_FILE); s = np.loadtxt(C.SDS_FILE)
    df = C.load_responses(latest_only=True)
    ids = df['experiment_id'].astype(str)
    lang = ids.map({k: v.get('language', 'EN') for k, v in exp.items()}).fillna('EN')
    pers = ids.map({k: v.get('persona', v.get('language', 'EN').lower()) for k, v in exp.items()}).fillna('en')
    label = ids.map({k: v.get('label', '') for k, v in exp.items()})
    vendor_of = {v.get('label', ''): v.get('vendor', 'Other') for v in exp.values()}

    def means_by_base(mask, strip):
        d = df[mask].copy(); d['base'] = label[mask].str.replace(strip, '', regex=False)
        return d.groupby('base')[V].mean()

    en = means_by_base((lang == 'EN') & (pers == 'en'), '')
    no = means_by_base((lang == 'NO') & (pers == 'no'), ' (NO)')
    bases = [b for b in no.index if b in en.index]

    # contributions: {axis: DataFrame [base x question]}
    cx, cy = {}, {}
    for b in bases:
        dz = (no.loc[b].values - en.loc[b].values) / s
        cx[b] = 1.81 * w[:, 0] * dz
        cy[b] = 1.61 * w[:, 1] * dz
    CX = pd.DataFrame(cx, index=V).T   # rows=models, cols=questions
    CY = pd.DataFrame(cy, index=V).T

    order = CX.mean().sort_values().index.tolist()   # questions by mean Δx contribution
    ypos = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for ax, (CC, name) in zip(axes, [(CX, 'Survival → Self-expression  (Δx)'),
                                     (CY, 'Traditional → Secular  (Δy)')]):
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=MUTED, lw=1, zorder=1)
        ax.barh(ypos, [CC[q].mean() for q in order], color=BAR, height=0.66, zorder=2,
                edgecolor='none')
        for b in bases:
            col = VENDOR_COLOR.get(vendor_of.get(b, 'Other'), '#6c757d')
            jit = (rng.random(len(order)) - 0.5) * 0.34
            ax.scatter([CC.loc[b, q] for q in order], ypos + jit, s=26, c=col,
                       edgecolors='white', linewidths=0.5, alpha=0.9, zorder=4)
        ax.set_title(name, fontsize=11.5, color=INK)
        ax.grid(axis='x', color=GRID, lw=0.6, alpha=0.6, zorder=0)
        ax.tick_params(colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([SHORT[q] for q in order], fontsize=10, color=INK)
    axes[0].set_xlabel('contribution to shift (map units)', fontsize=10, color='#52514e')
    axes[1].set_xlabel('contribution to shift (map units)', fontsize=10, color='#52514e')

    handles = ([Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=9, label=v)
                for v, c in VENDOR_COLOR.items()]
               + [Line2D([0], [0], marker='s', color='w', markerfacecolor=BAR, markersize=11,
                         label=f'mean of {len(bases)} models')])
    axes[1].legend(handles=handles, loc='lower right', fontsize=9, frameon=True,
                   facecolor=SURFACE, edgecolor=GRID)

    fig.suptitle('What drives the English→Norwegian shift, per question', fontsize=13.5, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = C.ROOT / 'out' / 'no_contrib.png'
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    print(f'Wrote {out}  ({len(bases)} models)')


if __name__ == '__main__':
    main()
