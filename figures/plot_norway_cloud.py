#!/usr/bin/env python3
"""
Paper figures 2 & 3: the cultural map with EVERY Norwegian survey respondent projected
onto it (thousands of faint black points), and the same again with a concentration
ellipse around the cloud.

Fig 2 -> figures/cultural_map_norway.png
Fig 3 -> figures/cultural_map_norway_ellipse.png  (--conf, default 0.95)

Individual respondents are projected with the SAME committed weights/means/sds as the
country centroids, reusing compute_weights.load_ivs() for identical scoring. The result
is cached to figures/_norway_points.csv so re-rendering is instant.

The ellipse is a CONFIDENCE ellipse OF THE MEAN (Hotelling T², identical definition to
the LLM ellipses via compare.hotelling_ellipse), so it means the same thing as the
chatbot ellipses. Because Norway has ~3,700 respondents its mean is pinned down very
tightly, so a high contour (default 99.99%) is used just to make it visible.

Usage: uv run --with matplotlib python figures/plot_norway_cloud.py [--conf 0.9999]
"""
import argparse
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
sys.path.insert(0, str(FIG_DIR))               # for plot_cultural_map
sys.path.insert(0, str(FIG_DIR.parent / 'src'))  # for compute_weights, compare
import plot_cultural_map as base  # noqa: E402
import compute_weights as CW  # noqa: E402
import compare as C  # noqa: E402

NORWAY_S003 = 578
CACHE = FIG_DIR / '_norway_points.csv'
CLOUD_COLOR = '#111111'


def norway_points() -> pd.DataFrame:
    """Per-respondent (x, y) for Norway, projected with the committed artifacts. Cached."""
    if CACHE.exists():
        return pd.read_csv(CACHE)
    weights = np.loadtxt(base.ROOT / 'out' / 'weights.txt')
    means = np.loadtxt(base.ROOT / 'out' / 'column_means.txt')
    sds = np.loadtxt(base.ROOT / 'out' / 'column_sds.txt')
    dx = CW.load_ivs()
    nor = dx[dx['S003'] == NORWAY_S003]
    X = nor[CW.FEATURES].astype(float).values
    scores = ((X - means) / sds) @ weights
    pts = pd.DataFrame({'x': 1.81 * scores[:, 0] + 0.038,
                        'y': 1.61 * scores[:, 1] - 0.1}).dropna()
    pts.to_csv(CACHE, index=False)
    print(f'Projected {len(pts)} complete-case Norwegian respondents (cached to {CACHE.name})')
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--conf', type=float, default=0.999999,
                   help='confidence level for the mean ellipse (high, so it is visible)')
    args = p.parse_args()

    pts = norway_points()
    xy = pts[['x', 'y']].values
    n = len(xy)
    nor = base.load_countries().set_index('country').loc['Norway']
    nx, ny = float(nor['x']), float(nor['y'])

    def render(with_ellipse):
        fig, ax = plt.subplots(figsize=(11, 8.6), dpi=200)
        fig.patch.set_facecolor(base.SURFACE)
        extra = [Line2D([0], [0], marker='o', color='w', markerfacecolor=CLOUD_COLOR,
                        markeredgecolor='none', markersize=8, alpha=0.6,
                        label=f'Norwegian respondents (n={n:,})')]
        # cloud sits under the country dots' labels but is drawn before the base so the
        # coloured country dots stay readable on top.
        ax.scatter(xy[:, 0], xy[:, 1], s=6, c=CLOUD_COLOR, alpha=0.08, edgecolors='none', zorder=2)
        if with_ellipse:
            # confidence ellipse OF THE MEAN — same definition as the LLM ellipses.
            e = C.hotelling_ellipse(xy[:, 0], xy[:, 1], args.conf)
            ax.add_patch(Ellipse((nx, ny), 2 * e['ea'], 2 * e['eb'], angle=np.degrees(e['etheta']),
                                 facecolor='none', edgecolor=CLOUD_COLOR, lw=1, zorder=7))
            extra.append(Line2D([0], [0], color=CLOUD_COLOR, lw=1,
                                label=f'{args.conf*100:g}% mean conf. ellipse'))
        base.draw_base_map(ax, extra_handles=extra, labels=False)
        # highlight the Norway centroid exactly as in the general map: outlined dot + label
        ax.scatter([nx], [ny], s=46, c=base.GROUP_COLOR.get(nor['region'], '#888'),
                   edgecolors=base.INK, linewidths=0.9, zorder=8)
        ax.annotate('Norway', (nx, ny), fontsize=8.5, color=base.INK, zorder=8,
                    xytext=(5, 3), textcoords='offset points',
                    path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
        title = 'Norwegian respondents on the cultural map'
        if with_ellipse:
            title += f'  ({args.conf*100:g}% mean conf. ellipse)'
        ax.set_title(title, fontsize=15, color=base.INK, pad=12)
        fig.tight_layout()
        out = FIG_DIR / ('cultural_map_norway_ellipse.png' if with_ellipse else 'cultural_map_norway.png')
        fig.savefig(out, dpi=200, facecolor=base.SURFACE, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {out}')

    # render(with_ellipse=False)
    render(with_ellipse=True)


if __name__ == '__main__':
    main()
