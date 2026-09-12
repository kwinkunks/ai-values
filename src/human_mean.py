#!/usr/bin/env python3
"""
Compute the mean position of the human values on the Inglehart-Welzel map.

"Where does the average human plot?" — a check on whether the human centroid sits
at the origin. It doesn't quite: the projection is affine (compare.py:65-66),
    x = 1.81 * PC0 + 0.038,  y = 1.61 * PC1 - 0.1
so the properly-weighted pooled mean lands on the intercept (0.038, -0.1) — near,
but not at, the origin. If those two constants were 0 it would be at (0, 0).

This projects every EVS+WVS respondent with the committed PCA artifacts (no R
needed) and reports the pooled mean under two weighting schemes:
  * unweighted  — every respondent equal; == the intercept by construction
  * S017        — the WVS demographic weight (what compute_weights.py's PCA uses)
See https://www.worldvaluessurvey.org/WVSContents.jsp?CMSID=WEIGHT for the weights.
(A true population-weighted global mean would additionally scale by country
population, which isn't in the .dta files.)

Needs the licensed .dta files in data/ (same as compute_weights.py), but NOT R.

Usage:
  uv run python src/human_mean.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compute_weights import EVS_FILE, FEATURES, WVS_FILE, load_ivs

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'out'
WEIGHTS_FILE = OUT / 'weights.txt'
MEANS_FILE = OUT / 'column_means.txt'
SDS_FILE = OUT / 'column_sds.txt'

# Affine intercept from the Inglehart-Welzel rescaling (compare.py:65-66).
INTERCEPT = np.array([0.038, -0.1])


def main() -> None:
    for f in (EVS_FILE, WVS_FILE, WEIGHTS_FILE, MEANS_FILE, SDS_FILE):
        if not f.exists():
            sys.exit(f'Missing required file: {f}')

    weights = np.loadtxt(WEIGHTS_FILE)
    means = np.loadtxt(MEANS_FILE)
    sds = np.loadtxt(SDS_FILE)

    dx = load_ivs()
    X = dx[FEATURES].astype(float).values

    # Standardise and project. Missing items -> standardised 0 (== the column
    # mean), the pairwise-safe choice that leaves the pooled mean unbiased.
    z = (X - means) / sds
    scores = np.where(np.isnan(z), 0.0, z) @ weights
    coord = np.column_stack([1.81 * scores[:, 0] + 0.038,
                             1.61 * scores[:, 1] - 0.1])

    s017 = dx['S017'].astype(float).values
    schemes = {
        'unweighted': np.ones(len(dx)),
        'S017 (WVS weight)': s017,
    }

    print(f'{len(dx):,} respondents across {dx["S003"].nunique()} countries\n')
    print(f'{"weighting":30s} {"mean (x, y)":>20s}  {"|origin|":>9s}  {"|intercept|":>11s}')
    for name, w in schemes.items():
        mx, my = np.average(coord, axis=0, weights=w)
        print(f'{name:30s} {f"({mx:+.4f}, {my:+.4f})":>20s}  '
              f'{np.hypot(mx, my):>9.4f}  {np.hypot(mx - INTERCEPT[0], my - INTERCEPT[1]):>11.4f}')
    print(f'\nintercept (natural centre) = ({INTERCEPT[0]}, {INTERCEPT[1]})')


if __name__ == '__main__':
    main()
