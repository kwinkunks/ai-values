#!/usr/bin/env python3
"""
Role vs language: for a few models, plot the shift from the English baseline under
(a) an English "typical Norwegian person" persona (NO_ROLE) and (b) actually asking in
Norwegian (NO). Reuses compare.py's projection. Writes out/role_shift.png.

Usage: uv run --with matplotlib python src/plot_role.py
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
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).parent))
import compare as C  # noqa: E402

BASES = ['GPT-5.6 Sol', 'GPT-6 Astra', 'MAI Thinking 1']
MODEL_COLOR = {'GPT-5.6 Sol': '#2a78d6', 'GPT-6 Astra': '#eb6834', 'MAI Thinking 1': '#1baf7a'}
# condition key -> (question language, persona set, label suffix, marker)
COND = {
    'EN':      ('EN', 'en',          '',           'o'),
    'NO_ROLE': ('EN', 'en_no_role',  ' (NO role)', 's'),
    'NO':      ('NO', 'no',          ' (NO)',      'D'),
}

SURFACE, INK, MUTED, GRID, COUNTRY = '#fcfcfb', '#0b0b0b', '#898781', '#e1e0d9', '#c4c4c0'


def coords_by_condition():
    exp = json.load(open(C.EXPERIMENTS_FILE))
    w, m, s = (np.loadtxt(C.WEIGHTS_FILE), np.loadtxt(C.MEANS_FILE), np.loadtxt(C.SDS_FILE))
    df = C.load_responses(latest_only=True)
    ids = df['experiment_id'].astype(str)
    langs = ids.map({k: v.get('language', 'EN') for k, v in exp.items()}).fillna('EN')
    personas = ids.map({k: v.get('persona', v.get('language', 'EN').lower())
                        for k, v in exp.items()}).fillna('en')
    out = {}
    for cond, (lg, pers, suffix, _) in COND.items():
        sub = C.compute_llm_coordinates(df[(langs == lg) & (personas == pers)], exp, w, m, s)
        for b in BASES:
            row = sub[sub['country'] == b + suffix]
            if len(row):
                out[(b, cond)] = row.iloc[0]
    return out


def ellipse(ax, r, color):
    if pd.isna(r.ea):
        return
    ax.add_patch(Ellipse((r.x, r.y), 2 * r.ea, 2 * r.eb, angle=np.degrees(r.etheta),
                         facecolor=color, edgecolor='none', alpha=0.1, zorder=2))


def main():
    pts = coords_by_condition()
    countries = C.load_country_centroids()

    fig, ax = plt.subplots(figsize=(11, 9.5), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.axhline(0, color=GRID, lw=1, zorder=0); ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6, zorder=0)

    # backdrop
    ax.scatter(countries['x'], countries['y'], s=18, c=COUNTRY, alpha=0.6, edgecolors='none', zorder=1)
    nor = countries[countries['country'] == 'Norway']
    if len(nor):
        ax.scatter(nor['x'], nor['y'], marker='*', s=320, c='#52514e', edgecolors=INK, lw=1, zorder=6)
        ax.annotate('Norway', (float(nor.x.iloc[0]), float(nor.y.iloc[0])), fontsize=9,
                    fontweight='bold', color=INK, xytext=(8, 5), textcoords='offset points', zorder=7)
    ax.scatter([0.038], [-0.1], marker='+', s=140, c=INK, linewidths=2.5, zorder=6)

    for b in BASES:
        color = MODEL_COLOR[b]
        en = pts.get((b, 'EN'))
        if en is None:
            continue
        # ellipses + arrows from EN to each condition
        ellipse(ax, en, color)
        for cond, style in [('NO_ROLE', (0, (5, 3))), ('NO', 'solid')]:
            tgt = pts.get((b, cond))
            if tgt is None:
                continue
            ellipse(ax, tgt, color)
            ax.annotate('', xy=(tgt.x, tgt.y), xytext=(en.x, en.y),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.6, alpha=0.85,
                                        linestyle=style, shrinkA=7, shrinkB=7), zorder=3)
        # markers
        for cond, (_lg, _pers, _suffix, marker) in COND.items():
            r = pts.get((b, cond))
            if r is None:
                continue
            ax.scatter([r.x], [r.y], marker=marker, s=95, c=color, edgecolors='white', lw=1.2, zorder=5)
        ax.annotate(b, (en.x, en.y), fontsize=9, fontweight='bold', color=color,
                    xytext=(7, -12), textcoords='offset points', zorder=8)

    ax.set_xlabel('Survival  ←→  Self-expression', fontsize=11, color='#52514e')
    ax.set_ylabel('Traditional  ←→  Secular-rational', fontsize=11, color='#52514e')
    ax.set_title('Role vs language: English baseline → "Norwegian person" role (English) → Norwegian language',
                 fontsize=12.5, color=INK, pad=12)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_aspect('equal', 'box')

    model_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=b)
                     for b, c in MODEL_COLOR.items()]
    cond_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MUTED, markersize=10, label='English baseline'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=MUTED, markersize=10, label='Norwegian role (English)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=MUTED, markersize=9, label='Norwegian language'),
        Line2D([0], [0], color=MUTED, lw=1.6, linestyle=(0, (5, 3)), label='shift: role effect'),
        Line2D([0], [0], color=MUTED, lw=1.6, linestyle='solid', label='shift: language effect'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#52514e', markeredgecolor=INK, markersize=14, label='Norway'),
    ]
    leg1 = ax.legend(handles=model_handles, loc='upper left', fontsize=9, frameon=True,
                     facecolor=SURFACE, edgecolor=GRID, title='Model')
    ax.add_artist(leg1)
    ax.legend(handles=cond_handles, loc='lower right', fontsize=8.5, frameon=True,
              facecolor=SURFACE, edgecolor=GRID)

    fig.tight_layout()
    out = C.ROOT / 'out' / 'role_shift.png'
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    missing = [b for b in BASES if (b, 'NO_ROLE') not in pts]
    print(f'Wrote {out}' + (f'  (no NO_ROLE data for: {missing})' if missing else ''))


if __name__ == '__main__':
    main()
