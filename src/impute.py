"""
Missing-item handling for projecting respondents (people or LLMs) onto the cultural map.

A factor score needs all 10 items, so by default any respondent with a missing answer
gets no position (complete-case). With imputation, each missing standardised item is
replaced by its conditional mean given the respondent's observed items,

    z_missing = R[m, o] @ inv(R[o, o]) @ z_observed,

using the item correlation matrix R that the PCA itself factored (psych's `pr$r`,
saved by compute_weights.py as out/item_correlations.txt). This is single regression
imputation under a multivariate-normal model: it uses the respondent's own answers (not
the global mean, which would shrink them towards the map origin). Respondents with fewer
than `min_observed` items are still dropped.
"""
import numpy as np

MIN_OBSERVED = 6   # of 10 items; below this a position would be mostly imputed


def impute_standardised(Z: np.ndarray, R: np.ndarray, min_observed: int = MIN_OBSERVED) -> np.ndarray:
    """Fill NaNs in standardised data Z by conditional means under correlation R.
    Rows with fewer than `min_observed` observed items are left as all-NaN."""
    Z = Z.copy()
    miss = np.isnan(Z)
    n_obs = (~miss).sum(axis=1)
    Z[n_obs < min_observed] = np.nan
    todo = miss.any(axis=1) & (n_obs >= min_observed)
    # Rows sharing a missingness pattern share one regression, so solve per pattern.
    patterns, inverse = np.unique(miss[todo], axis=0, return_inverse=True)
    rows = np.flatnonzero(todo)
    for k, m in enumerate(patterns):
        o = ~m
        idx = rows[inverse.ravel() == k]
        B = np.linalg.solve(R[np.ix_(o, o)], R[np.ix_(o, m)])   # (n_obs x n_miss)
        Z[np.ix_(idx, m)] = Z[np.ix_(idx, o)] @ B
    return Z


def map_coords(X: np.ndarray, means, sds, weights, R: np.ndarray = None,
               min_observed: int = MIN_OBSERVED) -> np.ndarray:
    """Project raw item answers X (n x 10, NaN = missing) to Inglehart-Welzel (x, y).
    R=None is complete-case (any missing item -> NaN position); otherwise missing items
    are imputed under R first. The affine constants are the WVS official ones."""
    Z = (X - means) / sds
    if R is not None:
        Z = impute_standardised(Z, R, min_observed)
    scores = Z @ weights
    return np.c_[1.81 * scores[:, 0] + 0.038, 1.61 * scores[:, 1] - 0.1]
