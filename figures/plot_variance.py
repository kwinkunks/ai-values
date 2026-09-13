#!/usr/bin/env python3
"""
Paper figure: where the spread in an LLM's measured position comes from.
  Left  — per model, the spread from PROMPT VARIANTS (between-persona SD) vs from
          REPEAT RUNS (between-run SD). Almost every model sits well above the diagonal:
          changing the persona moves the position far more than re-running does.
  Right — consequence: the uncertainty in a model's mean position falls as SD/sqrt(n)
          with the number of prompt variants, so going from the original 10 to 30
          (the +20 variants) cuts it by ~42%.

Reads figures/_variance_components.csv (written by the analysis; see repo history).
Writes figures/variance_components.png.

Usage: uv run --with matplotlib python figures/plot_variance.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent
sys.path.insert(0, str(FIG_DIR))
sys.path.insert(0, str(FIG_DIR.parent / 'src'))
import plot_cultural_map as base  # noqa: E402  (shared style constants)
import numpy as np  # noqa: E402
import json  # noqa: E402
import compare as C  # noqa: E402

ACCENT = '#2a78d6'


def variance_components() -> pd.DataFrame:
    """Per model (with >=2 runs and >=2 personas): between-persona vs between-run variance
    of the projected position. Uses the filtered loader, so zero-shot runs are excluded."""
    V = C.VARIABLES
    exp = json.load(open(C.EXPERIMENTS_FILE))
    m, s, w = np.loadtxt(C.MEANS_FILE), np.loadtxt(C.SDS_FILE), np.loadtxt(C.WEIGHTS_FILE)
    df = C.load_responses(latest_only=False).dropna(subset=V)
    Z = ((df[V].values - m) / s) @ w
    df = df.assign(x=1.81 * Z[:, 0] + 0.038, y=1.61 * Z[:, 1] - 0.1)
    df['label'] = df['experiment_id'].astype(str).map({k: v['label'] for k, v in exp.items()})
    df['persona'] = df['system'].map(C._persona)

    def var2(g):
        return g['x'].var(ddof=1) + g['y'].var(ddof=1)
    rows = []
    for lab, g in df.groupby('label'):
        if g['experiment_id'].nunique() < 2 or g['persona'].nunique() < 2:
            continue
        pv = var2(g.groupby('persona')[['x', 'y']].mean())    # spread across persona means
        rv = var2(g.groupby('experiment_id')[['x', 'y']].mean())  # spread across run means
        rows.append((lab, pv, rv))
    R = pd.DataFrame(rows, columns=['label', 'persona_var', 'run_var'])
    R['persona_sd'] = np.sqrt(R['persona_var']); R['run_sd'] = np.sqrt(R['run_var'])
    return R


def main():
    R = variance_components()
    pv, rv = R['persona_var'].mean(), R['run_var'].mean()
    pct_persona = 100 * pv / (pv + rv)
    ratio = float((R['persona_sd'] / R['run_sd']).median())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.4), dpi=200)
    fig.patch.set_facecolor(base.SURFACE)

    # -- Panel A: prompt-variant spread vs repeat-run spread, per model --
    axA.set_facecolor(base.SURFACE)
    hi = max(R['persona_sd'].max(), R['run_sd'].max()) * 1.08
    axA.plot([0, hi], [0, hi], ls='--', color=base.MUTED, lw=1, zorder=1)
    axA.annotate('equal', (hi * 0.72, hi * 0.72), rotation=45, color=base.MUTED,
                 fontsize=8.5, ha='center', va='bottom')
    axA.scatter(R['run_sd'], R['persona_sd'], s=42, c=ACCENT, alpha=0.75,
                edgecolors='white', linewidths=0.6, zorder=3)
    axA.set_xlim(0, hi); axA.set_ylim(0, hi)
    axA.set_xlabel('Repeat-run spread  (SD of run means, map units)', fontsize=11, color=base.INK)
    axA.set_ylabel('Prompt-variant spread  (SD of persona means)', fontsize=11, color=base.INK)
    axA.set_title('The spread is from prompt variants, not repeat runs', fontsize=12.5, color=base.INK)
    axA.text(0.04, 0.96, f'{len(R)} models\nprompt variants ≈ {pct_persona:.0f}% of variance\n'
                         f'median {ratio:.1f}× the run spread',
             transform=axA.transAxes, va='top', ha='left', fontsize=10, color=base.INK,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=base.GRID))
    axA.grid(True, color=base.GRID, lw=0.5, alpha=0.6, zorder=0)

    # -- Panel B: uncertainty in the mean position vs number of prompt variants --
    axB.set_facecolor(base.SURFACE)
    n = np.arange(1, 41)
    se = np.sqrt(pv / n)                                   # SE of the mean ~ persona_SD / sqrt(n)
    axB.plot(n, se, color=ACCENT, lw=2.2, zorder=3)
    for k, lab in [(10, 'original 10'), (30, '30 (+20 variants)')]:
        y = np.sqrt(pv / k)
        axB.plot([k, k], [0, y], ls=':', color=base.MUTED, lw=1, zorder=2)
        axB.scatter([k], [y], s=55, c=base.INK, zorder=4)
        axB.annotate(f'{lab}\nSD={y:.2f}', (k, y), fontsize=9, color=base.INK,
                     xytext=(8, 6), textcoords='offset points')
    drop = 100 * (1 - np.sqrt(10 / 30))
    axB.annotate(f'+20 variants → −{drop:.0f}% uncertainty', xy=(30, np.sqrt(pv / 30)),
                 xytext=(35, 0.35), textcoords='data', fontsize=10, color=base.INK,
                 ha='right', arrowprops=dict(arrowstyle='->', color=base.MUTED))
    axB.set_xlim(0, 40); axB.set_ylim(0, np.sqrt(pv / 1) * 0.55)
    axB.set_xlabel('Number of prompt variants (personas)', fontsize=11, color=base.INK)
    axB.set_ylabel('Uncertainty in mean position  (SD, map units)', fontsize=11, color=base.INK)
    axB.set_title('More variants → a better-determined position', fontsize=12.5, color=base.INK)
    axB.grid(True, color=base.GRID, lw=0.5, alpha=0.6, zorder=0)

    for ax in (axA, axB):
        ax.tick_params(colors=base.MUTED)
        for sp in ax.spines.values():
            sp.set_color(base.GRID)

    fig.tight_layout()
    out = FIG_DIR / 'variance_components.png'
    fig.savefig(out, dpi=200, facecolor=base.SURFACE, bbox_inches='tight')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
