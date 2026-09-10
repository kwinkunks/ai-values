#!/usr/bin/env python3
"""
Build the twin-coords file for a language-comparison page (norsk.html, kartuli.html …).

Emits a country backdrop plus only the models that have BOTH an English run and a run
in the target `--language`, each model tagged with its `language` ('EN'/'<lang>') and a
shared `base` name, so the page can pair them and draw EN→<lang> shift vectors. Reuses
compare.py's projection so coordinates match index.html. Output global: window.TWIN_COORDS.

Usage:
  uv run python src/make_twin_coords.py --language NO   # -> out/coords_norsk.js
  uv run python src/make_twin_coords.py --language KA   # -> out/coords_kartuli.js
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import compare as C  # noqa: E402  (path set above)

# Output filename per language. Deliberately distinct from compare.py's coords_<lang>.js
# (which are plain window.COORDS LLM-only files). Falls back to coords_<lang>_twins.js.
SLUG = {'NO': 'norsk', 'KA': 'kartuli'}


def model_records(df_llm: pd.DataFrame, language: str) -> list[dict]:
    recs = []
    for r in df_llm.itertuples():
        rec = {'country': r.country, 'region': r.region,
               'x': round(r.x, 6), 'y': round(r.y, 6),
               'language': language, 'base': r.country.replace(f' ({language})', '')}
        if pd.notna(r.ea):
            rec.update(ea=round(r.ea, 6), eb=round(r.eb, 6),
                       etheta=round(r.etheta, 6), en=int(r.en))
        if isinstance(r.pts, list):
            rec['pts'] = r.pts
        recs.append(rec)
    return recs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--language', required=True, help="Target non-EN language code, e.g. NO or KA")
    args = p.parse_args()
    lang = args.language
    out_file = C.ROOT / 'out' / f'coords_{SLUG.get(lang, lang.lower() + "_twins")}.js'

    with open(C.EXPERIMENTS_FILE) as f:
        experiments = json.load(f)
    weights = np.loadtxt(C.WEIGHTS_FILE)
    means = np.loadtxt(C.MEANS_FILE)
    sds = np.loadtxt(C.SDS_FILE)

    df = C.load_responses(latest_only=True)
    langs = df['experiment_id'].astype(str).map(
        {k: v.get('language', 'EN') for k, v in experiments.items()}).fillna('EN')

    tgt_llm = C.compute_llm_coordinates(df[langs == lang], experiments, weights, means, sds)
    if tgt_llm.empty:
        sys.exit(f'No experiments found for language {lang!r} in responses.csv')
    twin_bases = {c.replace(f' ({lang})', '') for c in tgt_llm['country']}
    en_all = C.compute_llm_coordinates(df[langs == 'EN'], experiments, weights, means, sds)
    en_twins = en_all[en_all['country'].isin(twin_bases)]

    countries = C.load_country_centroids()
    country_recs = [{'country': r.country, 'region': r.region,
                     'x': round(r.x, 6), 'y': round(r.y, 6)}
                    for r in countries.itertuples()]

    data = {'countries': country_recs,
            'models': model_records(en_twins, 'EN') + model_records(tgt_llm, lang)}
    out_file.write_text('window.TWIN_COORDS = ' + json.dumps(data, indent=0) + ';\n')
    print(f'Wrote {out_file}: {len(country_recs)} countries, '
          f'{len(twin_bases)} twinned models (EN+{lang})')


if __name__ == '__main__':
    main()
