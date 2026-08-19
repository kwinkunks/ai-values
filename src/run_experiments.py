#!/usr/bin/env python3
"""
Run WVS survey questions against LLMs and append results to out/responses.csv.

Usage:
  python src/run_experiments.py --expts 1215 1216 1217
  python src/run_experiments.py --expts 1500
  python src/run_experiments.py --provider anthropic
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Allow running as `python src/run_experiments.py` from repo root.
sys.path.insert(0, str(Path(__file__).parent))
from convo import Convo
from score import compute_score

load_dotenv()

ROOT = Path(__file__).parent.parent
EXPERIMENTS_FILE = ROOT / 'config' / 'experiments.json'
QUESTIONS_FILE = ROOT / 'data' / 'Prompts_Questions.csv'
RESPONDENTS_FILE = ROOT / 'data' / 'Prompts_Respondent_Descriptors_General.csv'
RESPONSES_FILE = ROOT / 'out' / 'responses.csv'

VARIABLES = ['F063', 'Y003', 'F120', 'G006', 'E018', 'Y002', 'A008', 'F118', 'E025', 'A165']

# Appended to every system prompt to encourage terse, parseable answers.
SYSTEM_SUFFIX = '\n\nIt is very important to respond EXACTLY as requested. Be terse.'

# In non-zero-shot mode, re-ask a question up to this many times until the
# answer is short enough to parse (< 5 chars). The growing conversation
# history nudges the model toward brevity.
MAX_RETRIES = 5
SHORT_ANSWER_THRESHOLD = 5


def load_experiments() -> dict:
    with open(EXPERIMENTS_FILE) as f:
        return json.load(f)


def load_questions() -> dict[str, str]:
    """Return {VARIABLE: prompt_text} with the 'Question: ' prefix stripped."""
    df = pd.read_csv(QUESTIONS_FILE)
    questions = df.set_index('scale')['prompt'].to_dict()
    return {k.upper(): v[10:] for k, v in questions.items()}


def load_respondents() -> list[str]:
    return pd.read_csv(RESPONDENTS_FILE)['respondent_descriptor'].tolist()


def run_experiment(expt_id: str, cfg: dict, questions: dict, respondents: list,
                   position: int = 1) -> list[dict]:
    provider = cfg['provider']
    model = cfg['model']
    zero_shot = cfg.get('zero_shot', False)
    reasoning_effort = cfg.get('reasoning_effort')
    run_at = datetime.now(timezone.utc).isoformat(timespec='seconds')

    rows = []
    # Each concurrently-running experiment gets its own stacked progress bar
    # (position 0 is the top-level bar in main()), so they don't clobber each
    # other and you can see per-respondent progress within every experiment.
    for system in tqdm(respondents, desc=f'  {expt_id}', position=position, leave=False):
        system_prompt = system + SYSTEM_SUFFIX
        convo = None if zero_shot else Convo(provider, model, system_prompt)
        row = {'experiment_id': expt_id, 'run_at': run_at, 'system': system}

        for variable, question in questions.items():
            prompt = f'Question: {question}'

            if zero_shot:
                answer = Convo(provider, model, system_prompt).ask(prompt, reasoning_effort).casefold()
            else:
                for _ in range(MAX_RETRIES):
                    answer = convo.ask(prompt, reasoning_effort).casefold()
                    if len(answer) < SHORT_ANSWER_THRESHOLD:
                        break

            try:
                row[variable] = compute_score(variable, answer)
            except (ValueError, AttributeError):
                tqdm.write(f'  Could not score {variable} for expt {expt_id}: {answer!r}')
                row[variable] = np.nan

        rows.append(row)
    return rows


def append_responses(rows: list[dict]) -> None:
    df_new = pd.DataFrame(rows)
    RESPONSES_FILE.parent.mkdir(exist_ok=True)
    if RESPONSES_FILE.exists():
        df_out = pd.concat([pd.read_csv(RESPONSES_FILE), df_new], ignore_index=True)
    else:
        df_out = df_new
    # Scores are integers; store as nullable Int64 so a missing value in a
    # column doesn't coerce the whole column to float ("5.0" instead of "5").
    df_out[VARIABLES] = df_out[VARIABLES].astype('Int64')
    df_out.to_csv(RESPONSES_FILE, index=False)


def select_experiments(all_expts: dict, ids: list[str]) -> dict:
    """
    Return a dict of experiments to run, filtered by the given IDs (or all if None).
    """
    unknown = [i for i in ids if i not in all_expts]
    if unknown:
        print(f'Warning: unknown experiment IDs: {unknown}')
    return {k: all_expts[k] for k in ids if k in all_expts}


def already_run_ids() -> set[str]:
    """IDs already present in responses.csv, as strings (the column is stored as int)."""
    if not RESPONSES_FILE.exists():
        return set()
    existing = pd.read_csv(RESPONSES_FILE, usecols=['experiment_id'])['experiment_id']
    return {str(i) for i in existing.unique()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--expts', nargs='+', metavar='ID', help='Experiment IDs to run')
    parser.add_argument('--jobs', '-j', type=int, default=4,
                        help='Max experiments to run concurrently (default: 4)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run experiments even if they already appear in responses.csv')
    args = parser.parse_args()

    all_expts = load_experiments()
    to_run = select_experiments(all_expts, args.expts)

    if not to_run:
        print('No experiments matched.')
        return

    manual = [k for k, v in to_run.items() if v.get('manual')]
    if manual:
        print(f'Skipping manual experiments: {manual}')
    to_run = {k: v for k, v in to_run.items() if not v.get('manual')}

    if not args.force:
        done = already_run_ids()
        skipped = [k for k in to_run if k in done]
        if skipped:
            print(f'Skipping already-run experiments (use --force to re-run): {skipped}')
        to_run = {k: v for k, v in to_run.items() if k not in done}

    if not to_run:
        print('Nothing to run.')
        return

    questions = load_questions()
    respondents = load_respondents()

    # Experiments are independent and I/O-bound, so run them concurrently.
    # Completions are collected in this (main) thread, keeping all CSV writes
    # single-threaded — no locking needed, and progress is saved incrementally.
    workers = max(1, min(args.jobs, len(to_run)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_experiment, expt_id, cfg, questions, respondents, position=i + 1): expt_id
            for i, (expt_id, cfg) in enumerate(to_run.items())
        }
        for expt_id, cfg in to_run.items():
            tqdm.write(f'{expt_id}: {cfg["label"]}  ({cfg["provider"]}/{cfg["model"]})')

        for future in tqdm(as_completed(futures), total=len(futures), desc='Experiments', position=0):
            expt_id = futures[future]
            try:
                rows = future.result()
            except Exception as e:  # noqa: BLE001 - isolate one experiment's failure
                tqdm.write(f'  ✗ {expt_id} failed: {e!r}')
                continue
            append_responses(rows)
            tqdm.write(f'  → {expt_id}: {len(rows)} rows saved to {RESPONSES_FILE}')


if __name__ == '__main__':
    main()
