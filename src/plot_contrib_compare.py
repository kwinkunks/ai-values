#!/usr/bin/env python3
"""
Per-question contribution to the map shift, LANGUAGE effect vs ROLE effect, on the models
that have both an EN generic run, a NO generic run, and an EN Norwegian-role run.

For each model and question i, relative to the EN generic baseline:
  language effect: z from NO generic;  role effect: z from EN "Norwegian person" role.
Contribution to Δx = 1.81·w[i,0]·Δz, to Δy = 1.61·w[i,1]·Δz (sum to the model's Δx/Δy).
Bars = mean across models; dots = each model. Writes out/no_contrib_compare.png.

Usage: uv run --with matplotlib python src/plot_contrib_compare.py
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

SHORT = {'F063': 'God important', 'Y003': 'Child autonomy', 'F120': 'Abortion OK',
         'G006': 'National pride', 'E018': 'Respect authority', 'Y002': 'Country aims',
         'A008': 'Happiness', 'F118': 'Homosexuality OK', 'E025': 'Signed petition',
         'A165': 'Trust people'}
LANG_COL, ROLE_COL = '#2a78d6', '#eb6834'   # language = blue, role = orange
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'


def main():
    V = C.VARIABLES
    exp = json.load(open(C.EXPERIMENTS_FILE))
    w = np.loadtxt(C.WEIGHTS_FILE); s = np.loadtxt(C.SDS_FILE)
    df = C.load_responses(latest_only=True)
    ids = df['experiment_id'].astype(str)
    lang = ids.map({k: v.get('language', 'EN') for k, v in exp.items()}).fillna('EN')
    pers = ids.map({k: v.get('persona', v.get('language', 'EN').lower()) for k, v in exp.items()}).fillna('en')
    label = ids.map({k: v.get('label', '') for k, v in exp.items()})

    def means_by_base(mask, strip):
        d = df[mask].copy(); d['base'] = label[mask].str.replace(strip, '', regex=False)
        return d.groupby('base')[V].mean()

    en = means_by_base((lang == 'EN') & (pers == 'en'), '')
    no = means_by_base((lang == 'NO') & (pers == 'no'), ' (NO)')
    role = means_by_base((lang == 'EN') & (pers == 'en_no_role'), ' (NO role)')
    bases = [b for b in en.index if b in no.index and b in role.index]

    def contrib(tgt):   # {axis: DataFrame [model x question]}
        cx, cy = {}, {}
        for b in bases:
            dz = (tgt.loc[b].values - en.loc[b].values) / s
            cx[b] = 1.81 * w[:, 0] * dz
            cy[b] = 1.61 * w[:, 1] * dz
        return pd.DataFrame(cx, index=V).T, pd.DataFrame(cy, index=V).T
    Lx, Ly = contrib(no)
    Rx, Ry = contrib(role)

    order = Lx.mean().sort_values().index.tolist()
    y = np.arange(len(order)); off = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for ax, (L, R, name) in zip(axes, [(Lx, Rx, 'Survival → Self-expression  (Δx)'),
                                       (Ly, Ry, 'Traditional → Secular  (Δy)')]):
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=MUTED, lw=1, zorder=1)
        ax.barh(y + off, [L[q].mean() for q in order], color=LANG_COL, height=0.36, alpha=0.35, zorder=2)
        ax.barh(y - off, [R[q].mean() for q in order], color=ROLE_COL, height=0.36, alpha=0.35, zorder=2)
        for b in bases:
            ax.scatter([L.loc[b, q] for q in order], y + off + (rng.random(len(order)) - .5) * .22,
                       s=24, c=LANG_COL, edgecolors='white', linewidths=.5, zorder=4)
            ax.scatter([R.loc[b, q] for q in order], y - off + (rng.random(len(order)) - .5) * .22,
                       s=24, c=ROLE_COL, edgecolors='white', linewidths=.5, zorder=4)
        ax.set_title(name, fontsize=11.5, color=INK)
        ax.grid(axis='x', color=GRID, lw=0.6, alpha=0.6, zorder=0)
        ax.tick_params(colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_xlabel('contribution to shift (map units)', fontsize=10, color='#52514e')
    axes[0].set_yticks(y); axes[0].set_yticklabels([SHORT[q] for q in order], fontsize=10, color=INK)

    handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=LANG_COL, markersize=11, label='language effect (EN→NO)'),
               Line2D([0], [0], marker='s', color='w', markerfacecolor=ROLE_COL, markersize=11, label='role effect (EN→NO role)')]
    axes[1].legend(handles=handles, loc='lower right', fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    fig.suptitle(f'Per-question contribution: language vs role effect  (mean of {len(bases)} models: {", ".join(bases)})',
                 fontsize=12.5, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = C.ROOT / 'out' / 'no_contrib_compare.png'
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    print(f'Wrote {out}  ({len(bases)} models: {bases})')


if __name__ == '__main__':
    main()
