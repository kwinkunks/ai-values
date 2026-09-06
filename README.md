# Values

A one-page app showing how LLMs plot on the [Inglehart–Welzel Cultural Map](https://en.wikipedia.org/wiki/Inglehart%E2%80%93Welzel_cultural_map_of_the_world) — based on recent responses to 10 questions from the World Values Survey (WVS) and European Values Study (EVS). The axes are *Survival vs Self-expression* (x) and *Traditional vs Secular* (y).

[The app has no server and shares no data, try it right now.](https://kwinkunks.github.io/ai-values)

<img height="800" alt="image" src="https://github.com/user-attachments/assets/82d68df3-6949-4264-a3a1-083dcb60cae4" />

## Setup

This is a [`uv`](https://docs.astral.sh/uv/) project: clone the repo and run `uv sync`. API keys go in a `.env` at the repo root, named `{PROVIDER}_API_KEY` (e.g. `ANTHROPIC_API_KEY`). Recomputing the PCA (step 0 below) additionally needs R — the `psych` package installs itself on first run via `rpy2` — plus the licensed WVS/EVS `.dta` files in `data/`; those licenses prevent me from sharing the data here.

## Run the analysis

Add or choose some experiments in `config/experiments.json`, then:

```bash
# 0. One-time (only if recomputing the PCA). Needs R + `psych`
#    (auto-installed on first run via rpy2). The committed out/*.txt
#    already has what you need.
uv run python src/compute_weights.py

# 1. Interview the LLMs. Appends rows to out/responses.csv (non-destructive).
uv run python src/run_experiments.py --expts 1543  # specific experiment id(s)

# 2. Project responses into Inglehart–Welzel space and write the plot data.
uv run python src/compare.py --out out/coords.js

# 3. View the map (index.html loads out/coords.js; no server needed).
open index.html
```

`compare.py` also accepts `--llm-only` (skip country centroids), `--all-runs` (average every run, not just the latest per experiment), and `--out FILE.csv`/stdout for CSV instead of the JS the app loads.

## How the pipeline works

**1. Interview → `out/responses.csv`.** `src/run_experiments.py` asks each configured LLM the 10 questions over the OpenAI-compatible chat API (`src/convo.py`). Answers are scored by `src/score.py`. Rows are appended with a `run_at` timestamp — non-destructive; each experiment is a one-off frozen snapshot (a new run = a new experiment id). Note that the questionnaire is done as a conversation, not 10 one-shot questions.

**2. PCA → `out/weights.txt`, `column_means.txt`, `column_sds.txt`, `country_centroids.csv`.** `src/compute_weights.py` combines EVS (waves 4 & 5) and WVS (waves 5–7), weights cases by **S017 · S018** (`--weight s017` for S017 only), and runs a 2-factor varimax PCA via R's `psych::principal()` (through `rpy2`). Missing codes → NaN (pairwise); `Y003` is cleaned to `[-2, 2]`; axes are assigned by content, so they can't come out swapped/mirrored regardless of psych's component order/sign. `data/s003.csv` maps WVS country codes to names/regions.

**3. Coordinates → `out/coords.js`.** `src/compare.py` projects the LLM scores into Inglehart–Welzel space using the saved weights and combines them with the country centroids → columns `country, region, x, y`.

### Coordinate transformation

```
x = 1.81 * PC0 + 0.038  # self-surv
y = 1.61 * PC1 - 0.1.   # trad-rat
```

PC0/PC1 are the dot product of standardized scores with the 10×2 PCA weight matrix. Note that Tao et al.'s R code uses the wrong parameters for this transformation (0.38 and -0.01). I believe this to be the authoritative source: https://worldvaluessurvey.org/WVSContents.jsp?CMSID=tradrat

## Adding a new model

1. Add entries to `config/experiments.json`. I usually add 3 runs for each model (each 'run' is 10 rounds of the 10 questions).
2. Run `run_experiments.py --expts <ids>`, then `compare.py --out out/coords.js`, then reload `index.html`.

The experiments are groups into 1000's (you'll see when you look at them). Required fields: `provider`, `model`, `label` (display name shown on the plot), `vendor` (the model's origin, as opposed to the API provider; used for looking up the endpoint). Optional: `release` (YYYY-MM), `reasoning_effort` (passed to the API), `zero_shot` (bool, default false), `manual` (bool, skips the CLI runner). Label convention: add an effort suffix (e.g. `GPT-5.5 (low)`) only to separate a model run at more than one reasoning effort.

## Scoring the 10 questions

The 10 IVS variables (F063, Y003, F120, G006, E018, Y002, A008, F118, E025, A165) are scored in `src/score.py` (Python) and mirrored in `computeScore()` / `compute_y002()` / `compute_y003()` in `index.html` (JavaScript) for the interactive quiz. Y002 and Y003 have non-trivial multi-choice scoring — keep the two implementations in sync.

## Further information

See [`paper.md`](./paper.md) for more on this project.
