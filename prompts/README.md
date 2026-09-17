# Hosted-LLM prompts

These are the system prompts the `llm` backend sends to a hosted LLM
(Azure OpenAI / OpenAI / Anthropic) for each task. They are
**versioned**: the loader looks for `prompts/<task>_<version>.txt`
(e.g. `prompts/category_v1.txt`).

## Available versions

| file | version | when to use |
|---|---|---|
| `category_v1.txt` | v1 | Default. 3 hand-written few-shot examples covering 3 categories. Works but the LLM has to guess for the other 15 categories. |
| `category_v2.txt` | v2 | Static balanced 18-example pool (one per category from `EXEMPLARS`). Every category is shown at least once in the prompt. Strict improvement over v1. |
| `category_v2.txt` + `exemplar_query=<batch[0]>` | v2-retrieved | Same prompt template, but the few-shot block is replaced with the K most similar exemplars from the pool for each batch's first comment. **Best expected accuracy on diverse inputs**, at the cost of slightly larger prompts and per-batch prompt assembly. |

The LLM backend auto-uses retrieval for `category` tasks at v2+
(`batched_classify(... prompt_version="v2")`). Pass `prompt_version="v1"`
or `"v2"` without a query to get the static block.

## Why versioned

LLM outputs drift when you change the prompt. By keeping each
version in a file under git, you can:

1. Roll back to a known-good prompt without code changes.
2. Diff prompts across releases.
3. A/B-test prompt v2 against v1 by setting
   `llm_options={"prompt_version": "v2"}`.

## Placeholders

Both prompt files use `{category_list}` and `{exemplar_block}`:

- `{category_list}` is rendered from `employee_voice.config.CATEGORIES`
  so the prompt can't drift from the runtime taxonomy.
- `{exemplar_block}` is rendered from `employee_voice.exemplars`:
  - If `exemplar_query` is provided (set automatically by the LLM
    backend for category v2+), the block contains the K exemplars
    most similar to the query.
  - Otherwise the block contains a balanced 18-exemplar subset of
    the curated `EXEMPLARS` pool (one per category).

The `test_prompt_includes_taxonomy` test asserts every category in
`CATEGORIES` is mentioned in the rendered category prompt, which
guards against drift between `config.py` and the exemplar pool.

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
