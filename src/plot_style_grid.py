#!/usr/bin/env python3
"""
Two-panel (Norway | Georgia) view of how prompt STYLE moves each model on the
Inglehart-Welzel map, decomposing the shift from the English baseline into:
  - role (English)         : "You are a typical Norwegian/Georgian person" (English questions)
  - language-only          : questions in the native language, generic persona
  - native (language+role) : native-language questions AND a native-language national role

Auto-includes any model that has an EN baseline plus at least one condition for that
country (so it picks up new models like Claude Sonnet 5 as their runs land). Reuses
compare.py's projection. Writes out/style_grid.png.

Usage: uv run --with matplotlib python src/plot_style_grid.py
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

# Per country: the (question-language, persona) cell for each condition, and the label suffix.
COUNTRIES = {
    'Norway':  {'role':   ('EN', 'en_no_role', ' (NO role)'),
                'lang':   ('NO', 'no',         ' (NO)'),
                'native': ('NO', 'no_no_role', ' (NO native)')},
    'Georgia': {'role':   ('EN', 'en_ka_role', ' (KA role)'),
                'lang':   ('KA', 'ka',         ' (KA)'),
                'native': ('KA', 'ka_ka_role', ' (KA native)')},
}
# condition -> (marker, arrow linestyle, legend label)
COND_STYLE = {
    'base':   ('o', None,          'English baseline'),
    'role':   ('s', (0, (5, 3)),   'role (English)'),
    'lang':   ('D', 'solid',       'language only'),
    'native': ('^', (0, (1, 2)),   'native (language + role)'),
}
# Stable colours (dataviz slots 1-3); extra models fall back to later slots.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7', '#e34948']
SURFACE, INK, MUTED, GRID, COUNTRY = '#fcfcfb', '#0b0b0b', '#898781', '#e1e0d9', '#c4c4c0'


def main():
    exp = json.load(open(C.EXPERIMENTS_FILE))
    w, m, s = (np.loadtxt(C.WEIGHTS_FILE), np.loadtxt(C.MEANS_FILE), np.loadtxt(C.SDS_FILE))
    df = C.load_responses(latest_only=True)
    ids = df['experiment_id'].astype(str)
    langs = ids.map({k: v.get('language', 'EN') for k, v in exp.items()}).fillna('EN')
    personas = ids.map({k: v.get('persona', v.get('language', 'EN').lower()) for k, v in exp.items()}).fillna('en')

    def cell(lang, persona, suffix):
        sub = C.compute_llm_coordinates(df[(langs == lang) & (personas == persona)], exp, w, m, s)
        return {r.country[:-len(suffix)] if suffix else r.country: r for r in sub.itertuples()}

    baseline = cell('EN', 'en', '')
    countries = C.load_country_centroids()

    # Gather points per (country, model, condition) and the set of models to plot.
    data = {c: {'base': baseline} for c in COUNTRIES}
    models = set()
    for cname, conds in COUNTRIES.items():
        for cond, (lang, persona, suffix) in conds.items():
            data[cname][cond] = cell(lang, persona, suffix)
            if cond in ('role', 'native'):                  # models defined by the new conditions,
                models |= set(data[cname][cond])            # not the many language-only runs
    models = [b for b in baseline if b in models]           # need an EN baseline
    models.sort(key=lambda b: (baseline[b].region, b))
    color = {b: PALETTE[i % len(PALETTE)] for i, b in enumerate(models)}

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.6), dpi=150, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, cname in zip(axes, COUNTRIES):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=GRID, lw=1, zorder=0); ax.axvline(0, color=GRID, lw=1, zorder=0)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.6, zorder=0)
        ax.scatter(countries['x'], countries['y'], s=14, c=COUNTRY, alpha=0.55, edgecolors='none', zorder=1)
        hi = countries[countries['country'] == cname]
        if len(hi):
            ax.scatter(hi['x'], hi['y'], marker='*', s=300, c='#52514e', edgecolors=INK, lw=1, zorder=6)
            ax.annotate(cname, (float(hi.x.iloc[0]), float(hi.y.iloc[0])), fontsize=9, fontweight='bold',
                        color=INK, xytext=(7, 5), textcoords='offset points', zorder=7)
        ax.scatter([0.038], [-0.1], marker='+', s=120, c=INK, linewidths=2.2, zorder=6)

        for b in models:
            col = color[b]
            base = data[cname]['base'].get(b)
            if base is None:
                continue
            for cond in ('role', 'lang', 'native', 'base'):
                marker, ls, _ = COND_STYLE[cond]
                r = data[cname].get(cond, {}).get(b)
                if r is None:
                    continue
                if pd.notna(r.ea):
                    ax.add_patch(Ellipse((r.x, r.y), 2 * r.ea, 2 * r.eb, angle=np.degrees(r.etheta),
                                         facecolor=col, edgecolor='none', alpha=0.08, zorder=2))
                if cond != 'base':                                  # arrow from baseline
                    ax.annotate('', xy=(r.x, r.y), xytext=(base.x, base.y),
                                arrowprops=dict(arrowstyle='-|>', color=col, lw=1.5, alpha=0.85,
                                                linestyle=ls, shrinkA=6, shrinkB=6), zorder=3)
                ax.scatter([r.x], [r.y], marker=marker, s=85, c=col, edgecolors='white', lw=1.1, zorder=5)
            ax.annotate(b, (base.x, base.y), fontsize=8.5, fontweight='bold', color=col,
                        xytext=(6, -12), textcoords='offset points', zorder=8)

        ax.set_title(cname, fontsize=13, color=INK)
        ax.set_xlabel('Survival  ←→  Self-expression', fontsize=10.5, color='#52514e')
        ax.tick_params(colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_aspect('equal', 'box')
    axes[0].set_ylabel('Traditional  ←→  Secular-rational', fontsize=10.5, color='#52514e')

    model_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color[b], markersize=10, label=b)
                     for b in models]
    cond_handles = ([Line2D([0], [0], marker=COND_STYLE[c][0], color='w', markerfacecolor=MUTED,
                            markersize=10, label=COND_STYLE[c][2]) for c in ('base', 'role', 'lang', 'native')]
                    + [Line2D([0], [0], color=MUTED, lw=1.5, linestyle=COND_STYLE[c][1], label=f'shift: {COND_STYLE[c][2]}')
                       for c in ('role', 'lang', 'native')])
    leg1 = axes[1].legend(handles=model_handles, loc='lower right', fontsize=8.5, frameon=True,
                          facecolor=SURFACE, edgecolor=GRID, title='Model')
    axes[1].add_artist(leg1)
    axes[0].legend(handles=cond_handles, loc='lower right', fontsize=8, frameon=True,
                   facecolor=SURFACE, edgecolor=GRID)

    fig.suptitle('Prompt style vs. the map: English baseline → nationality role, language, and both (native)',
                 fontsize=13.5, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = C.ROOT / 'out' / 'style_grid.png'
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    print(f'Wrote {out}  (models: {", ".join(models)})')


if __name__ == '__main__':
    main()
