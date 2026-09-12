#!/usr/bin/env python3
"""
Render a PNG of the Inglehart-Welzel map: country centroids as a recessive
backdrop, plus selected LLMs. Built for the EN-vs-NO comparison — OpenAI and
Anthropic 2026 models in English, with the Norwegian-language runs drawn as
diamonds and joined to their English twins by a connector.

Generate the inputs first:
  uv run python src/compare.py            --llm-only --out out/coords_en.js
  uv run python src/compare.py --language NO --llm-only --out out/coords_no.js
then:
  uv run python src/plot_map.py --out out/map_no.png
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

ROOT = Path(__file__).parent.parent
CENTROIDS_FILE = ROOT / 'out' / 'country_centroids.csv'

# Palette (validated data-viz reference instance): vendor = hue, language = shape.
VENDOR_COLOR = {'OpenAI': '#2a78d6', 'Anthropic': '#eb6834'}
SURFACE   = '#fcfcfb'
INK       = '#0b0b0b'
SECONDARY = '#52514e'
MUTED     = '#898781'
GRID      = '#e1e0d9'
COUNTRY   = '#c3c2b7'

# A few well-known countries labelled to orient the reader; Norway is highlighted
# separately since it is the reason for the Norwegian-language runs.
ANCHORS = ['Sweden', 'United States', 'Japan', 'Russia', 'Germany', 'Great Britain']
HIGHLIGHT = 'Norway'


def load_coords(path: Path) -> list[dict]:
    txt = path.read_text().strip()
    return json.loads(re.sub(r'^window\.COORDS = |;\s*$', '', txt))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--en', default=str(ROOT / 'out' / 'coords_en.js'), help='EN llm-only coords JS')
    p.add_argument('--no', default=str(ROOT / 'out' / 'coords_no.js'), help='NO llm-only coords JS')
    p.add_argument('--out', default=str(ROOT / 'out' / 'map_no.png'))
    p.add_argument('--year', default='2026', help='Only EN models released in this year')
    args = p.parse_args()

    countries = pd.read_csv(CENTROIDS_FILE).rename(columns={'surv-self': 'x', 'trad-sec': 'y'})
    en = [r for r in load_coords(Path(args.en))
          if r['region'] in VENDOR_COLOR and str(r.get('release', '')).startswith(args.year)]
    no = [r for r in load_coords(Path(args.no)) if r['region'] in VENDOR_COLOR]
    en_by_label = {r['country']: r for r in en}
    twin_labels = {r['country'].replace(' (NO)', '') for r in no}  # the 4 EN models with NO runs

    fig, ax = plt.subplots(figsize=(13, 10), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Quadrant reference lines + hairline grid, kept recessive.
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6, zorder=0)

    # Country backdrop.
    ax.scatter(countries['x'], countries['y'], s=20, c=COUNTRY, alpha=0.7,
               edgecolors='none', zorder=1)
    for _, row in countries.iterrows():
        if row['country'] in ANCHORS:
            ax.annotate(row['country'], (row['x'], row['y']), fontsize=7.5, color=MUTED,
                        xytext=(4, 3), textcoords='offset points', zorder=2)

    # Norway highlighted.
    nor = countries[countries['country'] == HIGHLIGHT]
    if not nor.empty:
        nx, ny = float(nor['x'].iloc[0]), float(nor['y'].iloc[0])
        ax.scatter([nx], [ny], marker='*', s=340, c=SECONDARY, edgecolors=INK,
                   linewidths=1.0, zorder=6)
        ax.annotate(HIGHLIGHT, (nx, ny), fontsize=10, fontweight='bold', color=INK,
                    xytext=(8, 6), textcoords='offset points', zorder=7)

    # Attenuated confidence ellipses (of the mean) for just the 4 models that have
    # both an EN and a NO run — EN twins and their NO counterparts.
    def draw_ellipse(rec):
        if rec.get('ea') is None:
            return
        ax.add_patch(Ellipse((rec['x'], rec['y']), width=2 * rec['ea'], height=2 * rec['eb'],
                             angle=np.degrees(rec['etheta']), facecolor=VENDOR_COLOR[rec['region']],
                             edgecolor='none', alpha=0.1, zorder=2))
    for lab in twin_labels:
        if lab in en_by_label:
            draw_ellipse(en_by_label[lab])
    for r in no:
        draw_ellipse(r)

    # Connectors EN -> NO (drawn under the markers).
    for r in no:
        twin = en_by_label.get(r['country'].replace(' (NO)', ''))
        if twin:
            ax.annotate('', xy=(r['x'], r['y']), xytext=(twin['x'], twin['y']),
                        arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.2,
                                        alpha=0.8, shrinkA=6, shrinkB=6), zorder=3)

    # EN models: circles.
    for r in en:
        ax.scatter([r['x']], [r['y']], marker='o', s=70, c=VENDOR_COLOR[r['region']],
                   edgecolors='white', linewidths=1.0, zorder=5)
        ax.annotate(r['country'], (r['x'], r['y']), fontsize=7, color=SECONDARY,
                    xytext=(5, 3), textcoords='offset points', zorder=8)

    # NO models: diamonds.
    for r in no:
        ax.scatter([r['x']], [r['y']], marker='D', s=85, c=VENDOR_COLOR[r['region']],
                   edgecolors='white', linewidths=1.0, zorder=6)
        ax.annotate(r['country'].replace(' (NO)', ' (NO)'), (r['x'], r['y']), fontsize=7.5,
                    fontweight='bold', color=INK, xytext=(6, -10),
                    textcoords='offset points', zorder=9)

    ax.set_xlabel('Survival  ←→  Self-expression', fontsize=11, color=SECONDARY)
    ax.set_ylabel('Traditional  ←→  Secular-rational', fontsize=11, color=SECONDARY)
    ax.set_title('LLM values on the Inglehart–Welzel map: OpenAI & Anthropic 2026, '
                 'English vs Norwegian prompts', fontsize=13, color=INK, pad=14)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=VENDOR_COLOR['OpenAI'],
               markeredgecolor='white', markersize=10, label='OpenAI'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=VENDOR_COLOR['Anthropic'],
               markeredgecolor='white', markersize=10, label='Anthropic'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MUTED, markersize=10,
               label='English prompt'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=MUTED, markersize=9,
               label='Norwegian prompt'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=SECONDARY,
               markeredgecolor=INK, markersize=15, label='Norway'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COUNTRY, markersize=8,
               label='Country (WVS/EVS)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=9, frameon=True,
              facecolor=SURFACE, edgecolor=GRID, labelcolor=SECONDARY)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    print(f'Wrote {args.out}  ({len(en)} EN + {len(no)} NO models, {len(countries)} countries)')


if __name__ == '__main__':
    main()
