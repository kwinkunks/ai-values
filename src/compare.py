#!/usr/bin/env python3
"""
Project LLM survey responses into Inglehart-Welzel space and combine with country centroids.

Output columns: country, region, x, y  (matches index.html CSV format)

Usage:
  python src/compare.py                  # latest run per experiment (default)
  python src/compare.py --all-runs       # average across all runs
  python src/compare.py --out coords.csv # write to file instead of stdout
  python src/compare.py --llm-only       # skip country centroids
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
EXPERIMENTS_FILE = ROOT / 'config' / 'experiments.json'
RESPONSES_FILE = ROOT / 'out' / 'responses.csv'
CENTROIDS_FILE = ROOT / 'out' / 'country_centroids.csv'
WEIGHTS_FILE = ROOT / 'out' / 'weights.txt'
MEANS_FILE = ROOT / 'out' / 'column_means.txt'
SDS_FILE = ROOT / 'out' / 'column_sds.txt'

VARIABLES = ['F063', 'Y003', 'F120', 'G006', 'E018', 'Y002', 'A008', 'F118', 'E025', 'A165']


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


def compute_llm_coordinates(df: pd.DataFrame, experiments: dict, weights, means, sds) -> pd.DataFrame:
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

    result = df.groupby(['country', 'region'])[['x', 'y']].mean().reset_index()
    return result[['country', 'region', 'x', 'y']]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all-runs', action='store_true',
                        help='Average over all runs instead of using only the latest per experiment')
    parser.add_argument('--llm-only', action='store_true',
                        help='Output LLM coordinates only, without country centroids')
    parser.add_argument('--out', metavar='FILE',
                        help='Write output to FILE instead of stdout')
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
    llm_coords = compute_llm_coordinates(df, experiments, weights, means, sds)

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
        records = [{'country': r.country, 'region': r.region,
                    'x': round(r.x, 6), 'y': round(r.y, 6)}
                   for r in combined.itertuples()]
        js = 'window.COORDS = ' + json.dumps(records, indent=0) + ';\n'
        Path(args.out).write_text(js)
        print(f'Wrote {summary} to {args.out}')
    else:
        csv_text = combined.to_csv(index=False, float_format='%.10f')
        if args.out:
            Path(args.out).write_text(csv_text)
            print(f'Wrote {summary} to {args.out}')
        else:
            print(csv_text, end='')


if __name__ == '__main__':
    main()
