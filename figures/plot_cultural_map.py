#!/usr/bin/env python3
"""
Paper figure 1: the Inglehart-Welzel cultural map — country means only (no LLMs),
coloured by cultural group, with a spread of key countries labelled directly.
Reads out/country_centroids.csv. Writes figures/cultural_map.png.

`draw_base_map(ax)` is reused by the Norway-cloud figures (plot_norway_cloud.py).

Usage: uv run --with matplotlib python figures/plot_cultural_map.py
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent.parent
CENTROIDS = ROOT / 'out' / 'country_centroids.csv'
OUT = Path(__file__).parent / 'cultural_map.png'

# Cultural group -> colour (matplotlib tab10, distinct qualitative).
GROUP_COLOR = {
    'African-Islamic':   '#d62728',   # red
    'Latin America':     '#ff7f0e',   # orange
    'Catholic Europe':   '#1f77b4',   # blue
    'Orthodox Europe':   '#17becf',   # cyan
    'West & South Asia': '#e377c2',   # pink
    'Protestant Europe': '#2ca02c',   # green
    'English-Speaking':  '#9467bd',   # purple
    'Confucian':         '#8c564b',   # brown
}
# Direct labels: a spread of well-known countries across groups and quadrants.
LABELS = ['United States', 'Great Britain', 'Australia', 'Sweden', 'Norway', 'Germany',
          'France', 'Poland', 'Russia', 'Ukraine', 'Japan', 'China', 'South Korea',
          'India', 'Brazil', 'Mexico', 'Nigeria', 'Egypt', 'Turkey', 'Iran']
LABEL_LEFT = {'Nigeria', 'Iran', 'Ukraine', 'Great Britain'}   # label to the left of the dot
HUMAN_AVG = (0.038, -0.1)   # affine intercept = human population mean

INK, MUTED, GRID, SURFACE, GREY = '#0b0b0b', '#7a7a7a', '#e1e0d9', '#ffffff', '#b3b3b3'


def load_countries() -> pd.DataFrame:
    return pd.read_csv(CENTROIDS).rename(columns={'surv-self': 'x', 'trad-sec': 'y'})


def _label(ax, text, x, y, left=False):
    ax.annotate(text, (x, y), fontsize=8.5, color=INK, zorder=8,
                xytext=(-5 if left else 5, 3), textcoords='offset points',
                ha='right' if left else 'left',
                path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])


def draw_base_map(ax, extra_handles=None, labels=True, grey=False, label_countries=None, legend=True):
    """Draw the country cultural map on `ax` (dots, human-average marker, axes, legend).

    grey=False: colour countries by cultural group; with labels=True add the key-country
      direct labels + outlines and the 'human average' text.
    grey=True (the LLM-figure style): all countries grey except Norway (black, labelled);
      also label each country in `label_countries` (e.g. United States, China).
    `extra_handles` (e.g. chatbot legend entries) are appended to the legend.
    """
    df = load_countries()
    present = df.set_index('country')
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=GRID, lw=1, zorder=0); ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6, zorder=0)

    if grey:
        others = df[df['country'] != 'Norway']
        ax.scatter(others['x'], others['y'], s=40, c=GREY, edgecolors='none', zorder=3)
        for name in (label_countries or []):
            if name == 'Norway' or name not in present.index:
                continue
            r = present.loc[name]
            ax.scatter([r.x], [r.y], s=44, c=GREY, edgecolors=INK, linewidths=0.8, zorder=5)
            _label(ax, name, r.x, r.y, name in LABEL_LEFT)
        r = present.loc['Norway']
        ax.scatter([r.x], [r.y], s=48, c='k', edgecolors='k', zorder=6)
        _label(ax, 'Norway', r.x, r.y, 'Norway' in LABEL_LEFT)
        base_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=GREY, markeredgecolor='none',
                   markersize=9, label='Country (survey mean)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markeredgecolor='none',
                   markersize=9, label='Norway'),
        ]
        legend_title = None
    else:
        for g, col in GROUP_COLOR.items():
            sub = df[df['region'] == g]
            ax.scatter(sub['x'], sub['y'], s=46, c=col, edgecolors='none', alpha=0.95, zorder=3)
        if labels:
            for name in LABELS:
                if name not in present.index:
                    print(f'  (label not in data, skipped: {name})')
                    continue
                r = present.loc[name]
                ax.scatter([r.x], [r.y], s=46, c=GROUP_COLOR.get(r.region, '#888'),
                           edgecolors=INK, linewidths=0.9, zorder=5)
                _label(ax, name, r.x, r.y, name in LABEL_LEFT)
        base_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                               markeredgecolor='none', markersize=9, label=g)
                        for g, c in GROUP_COLOR.items()]
        legend_title = 'Cultural group'

    ax.scatter([HUMAN_AVG[0]], [HUMAN_AVG[1]], marker='+', s=150, c=INK, linewidths=2.2, zorder=6)
    if labels and not grey:
        _label(ax, 'human average', HUMAN_AVG[0], HUMAN_AVG[1])

    ax.set_xlabel('Survival  ↔  Self-expression', fontsize=12, color=INK)
    ax.set_ylabel('Traditional  ↔  Secular-rational', fontsize=12, color=INK)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_aspect('equal', 'box')
    ax.margins(0.04)

    if legend:
        handles = base_handles + (extra_handles or [])
        ax.legend(handles=handles, loc='lower right', fontsize=9, frameon=True, facecolor=SURFACE,
                  edgecolor=GRID, title=legend_title, title_fontsize=9.5)


def main():
    fig, ax = plt.subplots(figsize=(11, 8.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    draw_base_map(ax)
    ax.set_title('The Inglehart–Welzel Cultural Map', fontsize=15, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    print(f'Wrote {OUT}')


if __name__ == '__main__':
    main()
