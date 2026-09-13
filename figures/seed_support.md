# RNG seed support by provider

Best-effort as of September 2026 — provider APIs change often, so **verify against current
docs before relying on this**. All providers in `src/convo.py` are called through the
OpenAI-compatible client, and `Convo.ask()` currently passes only `model`, `messages`,
and optional `reasoning_effort`. No `seed` is sent at present, so nothing is seeded.

| Provider | `seed`? | Notes |
|---|---|---|
| OpenAI | ✅ | `seed` + `system_fingerprint`, explicitly best-effort |
| Foundry (Azure OpenAI) | ✅ | same as OpenAI |
| Mistral | ✅ | `random_seed` |
| Gemini (OpenAI-compat) | ✅ | `seed` in gen config; accepted via the compat layer |
| Fireworks | ✅ | OpenAI-compatible, `seed` supported |
| Qwen / DashScope | ✅ | `seed` supported |
| DeepSeek | ✅ (likely) | OpenAI-compatible; documented `seed` |
| xAI (Grok) | ❓ | OpenAI-compat; `seed` accepted but determinism unclear |
| HuggingFace router | ❓ | depends on the backend model/provider |
| Anthropic | ❌ | no seed parameter |
| Perplexity | ❌ | Sonar API doesn't expose one |

Roughly ~7 of 11 reliably, ~2 uncertain (xAI, HF), 2 without (Anthropic, Perplexity).

## "Best effort"

Even where a `seed` exists, none of these give *true* determinism — MoE routing, request
batching, mixed-precision, and silent server-side model updates all inject variation
(OpenAI/Mistral document it as best-effort, with a fingerprint that changes under you). A
seed *reduces* run-to-run scatter but won't eliminate it.

## Low priority

Run-to-run variance is already the small component: ~11% of position variance, a median
~3.6× smaller than the prompt-variant (persona) spread (see `variance_components.png`).
Seeding would tighten the thing that matters least, so it wouldn't materially change any
model's position or the paper's conclusions — it's mainly a reproducibility/audit nicety.
