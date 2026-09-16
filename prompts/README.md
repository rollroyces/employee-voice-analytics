# Hosted-LLM prompts

These are the system prompts the `llm` backend sends to a hosted LLM
(Azure OpenAI / OpenAI / Anthropic) for each task. They are
**versioned**: the loader looks for `prompts/<task>_<version>.txt`
(e.g. `prompts/category_v1.txt`).

## Why versioned

LLM outputs drift when you change the prompt. By keeping each
version in a file under git, you can:

1. Roll back to a known-good prompt without code changes.
2. Diff prompts across releases.
3. A/B-test prompt v2 against v1 by setting
   `llm_options={"prompt_version": "v2"}`.

## Placeholders

`prompts/category_v1.txt` uses `{category_list}`, which `load_prompt`
substitutes from `employee_voice.config.CATEGORIES` at load time.
This means if you edit `CATEGORIES` in `config.py`, the prompt
updates automatically — no risk of the prompt and the
post-processing taxonomy drifting apart.

The category-prompt test `tests/test_llm_backend.py::test_prompt_includes_taxonomy`
asserts every category in `CATEGORIES` is mentioned in the prompt.

## Authoring a new version

1. Copy the current `_v1.txt` to `_v2.txt`.
2. Edit.
3. In a test, validate the new prompt parses (regex-check the
   categories / labels are present, count tokens, etc.).
4. Run the existing accuracy bench (`benchmarks/bench_category_accuracy.py`)
   with `prompt_version="v2"` on a labelled eval set.
5. Promote to v1 by renaming.

## Prompt fields

| field | prompt file | what the model returns |
|---|---|---|
| `category` | `category_v1.txt` | `{"category": str, "secondary"?: str, "score"?: float}` |
| `sentiment` | `sentiment_v1.txt` | `{"label": "positive"|"negative"|"neutral", "score": float}` |
| `emotion` | `emotion_v1.txt` | `{"label": <one of 28 GoEmotions>, "score": float}` |

All three prompts instruct the model to respond with **only valid
JSON** — no markdown fencing, no commentary. The response is parsed
by `llm_backend._parse_response()`, which is robust to a leading
"```json" fence if the model slips up.

## Caveats

- The prompts assume English. For multilingual HR data you'd add a
  language prefix and a translation step.
- The category prompt uses `{category_list}` placeholder so it stays
  in sync with `config.CATEGORIES`. The sentiment and emotion prompts
  hard-code their label sets (3 labels, 28 labels respectively).
