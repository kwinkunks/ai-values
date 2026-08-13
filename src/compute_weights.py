#!/usr/bin/env python3
"""
Compute the PCA artifacts from the Integrated Values Survey (EVS + WVS).

Reads the two Stata files in `data/`, runs R's `psych::principal()` (via rpy2)
for the weighted 2-factor varimax PCA, and writes the four files in `out/`
consumed by `src/compare.py`:

    weights.txt  column_means.txt  column_sds.txt  country_centroids.csv

Reproduces Tao et al. (2024)'s workflow but fixes its errors:
  * Inglehart-Welzel intercepts +0.038 (x) and -0.1 (y) — the WVS official
    values; Tao's R code uses +0.38 / -0.01, a decimal-slip error;
  * content-based axis assignment (which component loads on which item cluster)
    with fixed sign anchors — robust to psych's arbitrary component order/sign,
    unlike Tao's positional assignment;
  * clean Y003 (valid components only, clamped to [-2, 2]) and pairwise missing
    (replace missing codes with NaN rather than dropping rows).

Case weighting is S017 * S018 by default (WVS demographic weight x the N=1000
country-size normalisation, so large samples don't dominate the PCA);
`--weight s017` selects S017 only. Requires R with `psych` (installed on first
run) and the licensed WVS/EVS .dta files in data/.

Usage:
  uv run python src/compute_weights.py
  uv run python src/compute_weights.py --weight s017
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / 'data'
OUT = ROOT / 'out'
EVS_FILE = DATA / 'ZA7503_v3-0-0.dta'
WVS_FILE = DATA / 'Trends_VS_1981_2022_stata_v4_0.dta'
S003_FILE = DATA / 's003.csv'

# 10 IVS variables in the order weights.txt rows are stored (== compare.py's
# VARIABLES). Varimax is order-invariant, so this matches SPSS's alphabetical
# order as a map; only the row order of weights.txt differs.
FEATURES = ['F063', 'Y003', 'F120', 'G006', 'E018', 'Y002', 'A008', 'F118', 'E025', 'A165']
META = ['S003', 'S017', 'S018', 'versn_w']   # S018 only used by --weight s017s018

# Item clusters (Inglehart-Welzel), used to assign components to axes by content.
SURV_ITEMS = {'Y002', 'A008', 'F118', 'E025', 'A165'}   # Survival vs Self-expression
TRAD_ITEMS = {'F063', 'Y003', 'F120', 'G006', 'E018'}   # Traditional vs Secular

EVS_WAVES = ['4.0.0 (2015-10-30)', '5.0.0 (2022-06-08)']
WVS_WAVES = ['WVS5 v.20180912', 'WVS6 v.20201117', 'WVS7 v.5.0']

CRAN = 'https://cloud.r-project.org'


def load_ivs() -> pd.DataFrame:
    """Load + filter EVS and WVS; return combined data with missing codes as NaN
    (pairwise handling — rows are NOT dropped)."""
    nine = [c for c in FEATURES if c != 'Y003']

    evs = pd.read_stata(EVS_FILE, convert_categoricals=False)
    # EVS has no precomputed Y003; derive it from the four child-quality items,
    # but only where all four are valid (>= 0), else missing.
    aut = evs[['A029', 'A039', 'A040', 'A042']]
    evs['Y003'] = np.where((aut >= 0).all(axis=1),
                           aut['A029'] + aut['A039'] - aut['A040'] - aut['A042'], np.nan)
    evs = _filter_waves(evs, EVS_WAVES, 'EVS')

    wvs = pd.read_stata(WVS_FILE, convert_categoricals=False)  # WVS has native Y003
    wvs = _filter_waves(wvs, WVS_WAVES, 'WVS')

    dx = pd.concat([evs[META + FEATURES], wvs[META + FEATURES]]).reset_index(drop=True)

    # Keep only rows with a usable positive case weight.
    dx = dx[dx['S017'].notna() & (dx['S017'] > 0)].reset_index(drop=True)

    # Missing codes -> NaN (pairwise; rows are not dropped). Nine items use
    # -9..-1 as missing; Y003 is valid only in [-2, 2].
    for c in nine:
        dx.loc[dx[c] < 0, c] = np.nan
    dx.loc[(dx['Y003'] < -2) | (dx['Y003'] > 2), 'Y003'] = np.nan

    print(f'Combined: {len(dx)} weighted respondents; '
          f'Y003 range now [{dx["Y003"].min()}, {dx["Y003"].max()}]')
    return dx


def _filter_waves(df: pd.DataFrame, waves: list[str], label: str) -> pd.DataFrame:
    counts = {w: int((df['versn_w'] == w).sum()) for w in waves}
    if missing := [w for w, n in counts.items() if n == 0]:
        sys.exit(f'{label}: wave filter matched no rows for {missing}. '
                 f'Available: {sorted(df["versn_w"].unique())}')
    print(f'{label} waves kept: ' + ', '.join(f'{w}={n}' for w, n in counts.items()))
    return df.loc[df['versn_w'].isin(waves)]


def principal_weights(X: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted 2-factor varimax PCA via R's psych::principal().
    Returns the 10x2 factor-score weight matrix (rows in FEATURES order)."""
    from rpy2.robjects import default_converter, globalenv, numpy2ri, r
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr, isinstalled

    if not isinstalled('psych'):
        print('Installing R package "psych" (one-time)...')
        importr('utils').install_packages('psych', repos=CRAN)
    r('suppressMessages(library(psych))')

    with localconverter(default_converter + numpy2ri.converter):
        globalenv['X_R'] = X
        globalenv['w_R'] = weight                     # n x p case-weight matrix
        r('X_R <- as.matrix(X_R); w_R <- as.matrix(w_R)')
        r('pr <- principal(X_R, nfactors=2, rotate="varimax", use="pairwise", weight=w_R)')
        return np.asarray(r('pr$weights'))


def assign_axes(weights: np.ndarray) -> np.ndarray:
    """Return weights with column 0 = Survival–Self-expression and column 1 =
    Traditional–Secular, oriented so self-expression / secular are positive.
    Content-based, so psych's arbitrary component order/sign can't produce a
    swapped or mirrored map."""
    idx = {f: i for i, f in enumerate(FEATURES)}

    def load(col, cluster):
        return sum(abs(weights[idx[f], col]) for f in cluster)

    surv_col = 0 if load(0, SURV_ITEMS) > load(0, TRAD_ITEMS) else 1
    W = weights[:, [surv_col, 1 - surv_col]].copy()
    # Sign anchors: F118 (homosexuality justifiable) positive on self-expression;
    # F063 (importance of God) negative on the secular axis.
    signs = np.array([1 if W[idx['F118'], 0] >= 0 else -1,
                      1 if W[idx['F063'], 1] <= 0 else -1])
    return W * signs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weight', choices=['s017', 's017s018'], default='s017s018',
                    help='Case weighting: S017*S018 (default; WVS N=1000 weight) or S017 only')
    args = ap.parse_args()

    for f in (EVS_FILE, WVS_FILE, S003_FILE):
        if not f.exists():
            sys.exit(f'Missing required data file: {f}')

    dx = load_ivs()
    X = dx[FEATURES].astype(float).values                 # contains NaN (missing)
    # psych wants an n x p weight matrix; every column is the case weight.
    w = dx['S017'] if args.weight == 's017' else dx['S017'] * dx['S018']
    weight = w.astype(float).values[:, None] * np.ones((len(dx), len(FEATURES)))
    print(f'Case weighting: {args.weight}')

    weights = assign_axes(principal_weights(X, weight))

    # Unweighted standardisation params (used to standardise both countries here
    # and LLMs in compare.py, so both share one projection).
    column_means = np.nanmean(X, axis=0)
    column_sds = np.nanstd(X, axis=0, ddof=1)

    # Inglehart-Welzel affine map, using the WVS official rescaling constants.
    # NB Tao et al. (2024)'s R code erroneously uses +0.38 / -0.01.

    # Scores by projection (identical to psych's factor scores; NaN-safe row-wise).
    scores = ((X - column_means) / column_sds) @ weights
    dx = dx.assign(**{'surv-self': 1.81 * scores[:, 0] + 0.038,
                      'trad-sec': 1.61 * scores[:, 1] - 0.1})

    cc = dx.groupby('S003')[['surv-self', 'trad-sec']].mean()
    s003 = pd.read_csv(S003_FILE).rename(
        columns={'s003': 'S003', 'country.territory': 'country', 'Category': 'region'})
    centroids = (s003.merge(cc, on='S003', how='inner')
                     [['country', 'region', 'surv-self', 'trad-sec']]
                     .sort_values('country').reset_index(drop=True))

    OUT.mkdir(exist_ok=True)
    np.savetxt(OUT / 'weights.txt', weights)
    np.savetxt(OUT / 'column_means.txt', column_means)
    np.savetxt(OUT / 'column_sds.txt', column_sds)
    centroids.to_csv(OUT / 'country_centroids.csv', index=False)
    print(f'Wrote weights/means/sds and {len(centroids)} country centroids to {OUT}/')


if __name__ == '__main__':
    main()
