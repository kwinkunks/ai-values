#!/usr/bin/env python3
"""
Project LLM survey responses into Inglehart-Welzel space and combine with country centroids.

Output columns: country, region, x, y  (matches index.html CSV format)

Usage:
  python src/compare.py                  # latest run per experiment (default)
  python src/compare.py --all-runs       # average across all runs
  python src/compare.py --out coords.csv # write to file instead of stdout
  python src/compare.py --llm-only       # skip country centroids
  python src/compare.py --language NO --out out/coords_no.js  # Norwegian-run results
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

ROOT = Path(__file__).parent.parent
EXPERIMENTS_FILE = ROOT / 'config' / 'experiments.json'
RESPONSES_FILE = ROOT / 'out' / 'responses.csv'
CENTROIDS_FILE = ROOT / 'out' / 'country_centroids.csv'
WEIGHTS_FILE = ROOT / 'out' / 'weights.txt'
MEANS_FILE = ROOT / 'out' / 'column_means.txt'
SDS_FILE = ROOT / 'out' / 'column_sds.txt'

VARIABLES = ['F063', 'Y003', 'F120', 'G006', 'E018', 'Y002', 'A008', 'F118', 'E025', 'A165']

# Some legacy rows stored the persona WITH run_experiments' appended terseness
# suffix and others without, which splits one persona into two grouping keys.
# Strip it so the ellipse counts personas correctly (the suffix is constant, so
# carries no persona information).
_SUFFIX_RE = re.compile(r'\s*It is very important to respond EXACTLY as requested\.\s*Be terse\.\s*$')


def _persona(system: str) -> str:
    return _SUFFIX_RE.sub('', system).strip()

# Fewest per-label points needed for a Hotelling ellipse (needs n-2 >= 1 F dof,
# but 4 is the practical floor for a non-silly 2x2 covariance).
MIN_ELLIPSE_N = 4


def hotelling_ellipse(x: np.ndarray, y: np.ndarray, conf: float) -> dict | None:
    """80%-style confidence ellipse *of the mean* of n (x, y) points, via
    Hotelling's T^2. Returns semi-axes (ea, eb), major-axis angle (etheta, rad)
    and n (en), or None if there aren't enough / the covariance is degenerate."""
    n = len(x)
    if n < MIN_ELLIPSE_N:
        return None
    S = np.cov(x, y, ddof=1)
    lam, vecs = np.linalg.eigh(S)               # ascending eigenvalues
    lam, vecs = lam[::-1], vecs[:, ::-1]         # major axis first
    if lam[1] <= 0:                              # collinear / degenerate spread
        return None
    # Radius^2 so the true mean lies inside with probability `conf`.
    c2 = 2 * (n - 1) / (n * (n - 2)) * f_dist.ppf(conf, 2, n - 2)
    a, b = np.sqrt(c2 * lam)
    theta = float(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return {'ea': a, 'eb': b, 'etheta': theta, 'en': int(n)}


def load_responses(latest_only: bool) -> pd.DataFrame:
    df = pd.read_csv(RESPONSES_FILE)
    if latest_only and 'run_at' in df.columns:
        # Keep only rows from the most recent run per experiment. Pre-timestamp
        # snapshots (migrated from the old Colab CSV, run_at NaN) are always kept
        # — each experiment is run exactly once, so there is nothing to dedupe.
        latest = df.groupby('experiment_id')['run_at'].transform('max')
        df = df[df['run_at'].isna() | (df['run_at'] == latest)]
    return df


def load_country_centroids() -> pd.DataFrame:
    """Load pre-computed country centroids and normalise column names to x, y."""
    df = pd.read_csv(CENTROIDS_FILE)
    return df.rename(columns={'surv-self': 'x', 'trad-sec': 'y'})


def compute_llm_coordinates(df: pd.DataFrame, experiments: dict, weights, means, sds,
                            conf: float = 0.80) -> pd.DataFrame:
    # `label` is the curated display name (unique per config, incl. effort suffix
    # where needed); `vendor` is the model's true origin, used as the plot region
    # so points colour by vendor. Runs sharing a label are averaged into one point.
    label_map = {k: v['label'] for k, v in experiments.items()}
    vendor_map = {k: v['vendor'] for k, v in experiments.items()}
    df = df.copy()
    ids = df['experiment_id'].astype(str)
    df['country'] = ids.map(label_map)
    df['region'] = ids.map(vendor_map)
    df = df.dropna(subset=['country'] + VARIABLES)

    X_scaled = (df[VARIABLES].values - means) / sds
    transformed = np.dot(X_scaled, weights)

    # Inglehart-Welzel affine scaling (from Tao et al / compute_weights notebook).
    df['x'] = 1.81 * transformed[:, 0] + 0.038
    df['y'] = 1.61 * transformed[:, 1] - 0.1

    grouped = df.groupby(['country', 'region'])
    result = grouped[['x', 'y']].mean().reset_index()

    # Per-label confidence ellipse of the mean, computed over the label's
    # PERSONA means (one point per respondent-descriptor). Variance analysis
    # showed ~68% of a model's spread is persona/prompt-sensitivity and only ~3%
    # is run-to-run, so the persona is the meaningful sampling unit: the ellipse
    # then reads as "how much rephrasing moves this model" (n = #personas ~10),
    # not a spurious n=30 of correlated draws. Countries are precomputed means
    # with no per-sample spread here, so only LLM labels get ellipse columns.
    ellipses = {}
    pts_map = {}   # raw per-response (x, y) points per label, for the faint scatter
    for name, g in grouped:
        pm = g.groupby(g['system'].map(_persona))[['x', 'y']].mean()
        ellipses[name] = hotelling_ellipse(pm['x'].values, pm['y'].values, conf)
        pts_map[name] = [[round(float(a), 4), round(float(b), 4)] for a, b in zip(g['x'], g['y'])]
    for col in ('ea', 'eb', 'etheta', 'en'):
        result[col] = [(e or {}).get(col, np.nan)
                       for e in map(ellipses.get, zip(result['country'], result['region']))]
    result['pts'] = [pts_map.get(key) for key in zip(result['country'], result['region'])]

    # Lineage grouping + release date (constant per label) for "join the dots".
    # A blank/absent lineage means the point is not connected to any line.
    lineage_map = {v['label']: v.get('lineage') for v in experiments.values()}
    release_map = {v['label']: v.get('release') for v in experiments.values()}
    result['lineage'] = result['country'].map(lineage_map)
    result['release'] = result['country'].map(release_map)

    return result[['country', 'region', 'x', 'y', 'ea', 'eb', 'etheta', 'en',
                   'lineage', 'release', 'pts']]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all-runs', action='store_true',
                        help='Average over all runs instead of using only the latest per experiment')
    parser.add_argument('--llm-only', action='store_true',
                        help='Output LLM coordinates only, without country centroids')
    parser.add_argument('--out', metavar='FILE',
                        help='Write output to FILE instead of stdout')
    parser.add_argument('--conf', type=float, default=0.80, metavar='C',
                        help="Confidence level for each LLM's mean ellipse (default: 0.80)")
    parser.add_argument('--language', default='EN', metavar='LANG',
                        help="Only include experiments run in this language, per their "
                             "config `language` field (default: EN; absent field ⇒ EN)")
    args = parser.parse_args()

    required = [EXPERIMENTS_FILE, RESPONSES_FILE, WEIGHTS_FILE, MEANS_FILE, SDS_FILE]
    if not args.llm_only:
        required.append(CENTROIDS_FILE)
    for path in required:
        if not path.exists():
            sys.exit(f'Missing required file: {path}')

    with open(EXPERIMENTS_FILE) as f:
        experiments = json.load(f)

    weights = np.loadtxt(WEIGHTS_FILE)
    means = np.loadtxt(MEANS_FILE)
    sds = np.loadtxt(SDS_FILE)

    df = load_responses(latest_only=not args.all_runs)

    # Keep only experiments run in the requested language. Language is a property of
    # the experiment (config), not stored per response row; IDs missing from the
    # config (legacy rows) default to EN.
    lang_map = {k: v.get('language', 'EN') for k, v in experiments.items()}
    df = df[df['experiment_id'].astype(str).map(lang_map).fillna('EN') == args.language]

    llm_coords = compute_llm_coordinates(df, experiments, weights, means, sds, conf=args.conf)

    if args.llm_only:
        combined = llm_coords
    else:
        countries = load_country_centroids()
        combined = pd.concat([countries, llm_coords], ignore_index=True)

    summary = (f'{len(combined)} rows ({len(llm_coords)} LLMs'
               + (f', {len(combined) - len(llm_coords)} countries' if not args.llm_only else '')
               + ')')

    if args.out and args.out.endswith('.js'):
        # Emit a JS file that index.html loads via <script src> — works with
        # `open index.html` (file://), where fetch() of a .csv is blocked.
        records = []
        for r in combined.itertuples():
            rec = {'country': r.country, 'region': r.region,
                   'x': round(r.x, 6), 'y': round(r.y, 6)}
            # Ellipse fields only for LLM rows (countries have NaN).
            if pd.notna(r.ea):
                rec.update(ea=round(r.ea, 6), eb=round(r.eb, 6),
                           etheta=round(r.etheta, 6), en=int(r.en))
            # Lineage/release only where set (LLM rows with a non-blank lineage).
            if pd.notna(r.lineage) and r.lineage:
                rec['lineage'] = r.lineage
            if pd.notna(r.release):
                rec['release'] = r.release
            # Raw per-response points for the faint scatter (LLM rows only).
            if isinstance(r.pts, list):
                rec['pts'] = r.pts
            records.append(rec)
        js = 'window.COORDS = ' + json.dumps(records, indent=0) + ';\n'
        Path(args.out).write_text(js)
        print(f'Wrote {summary} to {args.out}')
    else:
        csv_text = combined.drop(columns=['pts']).to_csv(index=False, float_format='%.10f')
        if args.out:
            Path(args.out).write_text(csv_text)
            print(f'Wrote {summary} to {args.out}')
        else:
            print(csv_text, end='')


if __name__ == '__main__':
    main()
