<!--
Working note (not part of the rendered paper — the leading underscore makes Quarto
ignore it). Numbers computed from out/coords.js on 2026-09-24 by Claude, to ground the
Results section. Regenerate with the scripts in /tmp or re-derive from coords.js.
-->

# Results grounding — numbers behind the claims

## Reference points (human survey data)

- Human country cloud (n=107): mean = **(−0.055, −0.211)**; the affine intercept /
  weighted global mean ≈ **(0.038, −0.1)** (the `+` on the plots).
- United States = **(1.22, −0.40)**.
- Outlier thresholds vs the human country cloud (Mahalanobis, χ²₂): 95% → 2.45, 99% → 3.03.

## B2 — Drift over time is NOT uniform (⚠️ abstract needs qualifying)

Per-lineage drift vector (earliest→latest release, Δx = self-expression, Δy = secular):

| Lineage | first → last | Δ(x, y) | direction |
|---|---|---|---|
| Claude Sonnet | 3.7 → 5 | (−1.64, −0.87) | **toward origin/US** |
| Claude Opus | 4 → 5 | (−1.08, −0.22) | **toward origin/US** |
| Gemini | 2 → 3.8 Flash | (−0.83, −0.67) | **toward origin** |
| Mistral (S/L) | — | ≈(+0.3, −0.2) | slightly out (x), down (y) |
| GLM | 5 → 5.3 | (+0.63, −0.08) | outward (x) |
| Grok | 3 → 4.6 | (+0.34, +1.17) | **away — more secular** |
| GPT-5 | 5 → 5.5 | (+1.40, +0.79) | **away — more secular + self-exp** |
| Qwen | 3 → 3.7 | (+0.62, +2.40) | **away — much more secular** |
| DeepSeek | 3 → 4 | (−0.51, +0.57) | mixed |

**Key point for the author:** the abstract says "*most* models have generally been moving
towards the United States, but also towards the origin … away from the more secular /
self-expression values they expressed in 2024." That is true for **Anthropic (Claude) and
Google (Gemini)**, but the **opposite** is true for **OpenAI GPT-5, xAI Grok, and Alibaba
Qwen**, which have moved *up and out* (more secular-rational and/or more self-expression).
The honest headline is a **divergence**: some labs are converging toward the human centre,
others are drifting further into the secular/self-expression corner. Recommend rewording
the abstract and building B2 around this split rather than a single trend.

## B3 — Named "outliers": only some hold up as outliers *from humans*

| Model (as named in abstract) | position | Mahalanobis | outlier from humans? |
|---|---|---|---|
| Grok 4.1 | (1.99, 2.40) | 3.50 | ✅ yes (>99%) |
| Grok 4.1 (low) | (2.69, 3.92) | 5.50 | ✅ strong |
| GPT-5.1 | (3.81, 1.67) | 3.47 | ✅ yes (>99%) |
| Qwen 3 | (1.15, −1.74) | 2.71 | ⚠️ borderline (>95%, <99%) |
| Gemini 3 | (−0.80, −0.38) | 0.60 | ❌ **no — sits inside the human cloud** |
| Claude Fable 5 | (0.25, 0.48) | 0.92 | ❌ **no — near the human centre** |

**Two different senses of "outlier."** Grok 4.1 and GPT-5.1 are outliers *from the human
distribution* (extreme secular/self-expression). Gemini 3 and Claude Fable 5 are the
opposite — they are notable precisely because they sit unusually *close* to the human
average, i.e. outliers *among LLMs*. The abstract currently lumps both kinds together as
"statistical outliers in the global context," which is misleading. Recommend splitting the
sentence into (a) models far outside the human range and (b) models unusually central for an
LLM. (Also worth naming the true x-extremes not in the abstract: MS Copilot GPT-5 at
x=3.79, GPT-4 at 2.77, Claude Haiku 4.5 at 3.52.)

## B4 — Reasoning level (GPT) and size (Claude): real but non-monotonic

Reasoning effort (same base model, varying effort):
- GPT-5: full (1.19, 0.50) vs **minimal** (2.56, 0.02) — less reasoning → much more
  self-expression, less secular.
- GPT-5.1: full (3.81, 1.67) vs low (3.68, 1.08); GPT-5.5: full (2.59, 1.29) vs low
  (2.66, 0.90) vs none (2.45, 0.99) — more reasoning tends to raise the secular axis, but
  the size/direction varies by generation. Supports "not straightforward."

Model size (Claude, same generation ≈ 4.5):
- Haiku 4.5 (3.52, 0.33) ≫ Opus 4.5 (2.14, 0.51) > Sonnet 4.5 (1.50, 0.55) on
  self-expression. Ordering is **not** monotonic in size (Sonnet sits between Haiku and
  Opus by size but lowest on x). Supports "not straightforward."

## B5 — Robustness (already have figures)

- `variance_components.png`: prompt/persona variants ≈ 89% of variance, median 3.6× the
  repeat-run spread; +20 personas → −42% uncertainty in mean position (SD 0.39 → 0.23).
- `zeroshot_vs_conversation.png`: the conversation-vs-zero-shot choice shifts positions —
  supports reporting them as distinct conditions (zero-shot is excluded from the main map).

## Caveat

`out/coords.js` includes models newer than those the abstract names (e.g. Opus 5, GPT-6
Astra, Grok 4.6, Qwen 3.7). If the abstract was written against an earlier snapshot, some
discrepancies above may reflect that. Re-confirm the abstract against the final dataset
before submission.
