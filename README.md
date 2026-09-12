# Employee Voice Analytics

Reusable, dual-runtime (Local CLI + Databricks Spark) toolkit that turns
unstructured employee feedback (exit interviews, annual surveys, pulse
checks, Glassdoor comments) into structured HR signals:

1. **Multi-aspect sentiment + GoEmotions taxonomy** — positive /
   neutral / negative, then 28 fine-grained emotions, then per-aspect
   (Compensation, Manager, Team, Workload, Growth, Tools, Culture,
   WorkLifeBalance).
2. **HR taxonomy classification** — zero-shot against a configurable
   taxonomy (Compensation, Leadership, Workload, DE&I, Career Growth,
   Tools, …).
3. **Dynamic thematic discovery** — BERTopic / KeyBERT.
4. **Qualitative flight-risk indicators** — push / pull factor flags,
   q-over-q risk velocity, and a 0-100 RiskScore + RiskBand.
5. **Enterprise privacy guardrails** — PII scrubber (regex or
   Presidio) and k-anonymity suppression for under-threshold slices.

Dual-licensed: AGPL-3.0-or-later + commercial — see
[`LICENSE`](./LICENSE) and [`COMMERCIAL_LICENSE.md`](./COMMERCIAL_LICENSE.md).

## Architecture

```
                              ┌──────────────────────────────────────────────┐
                              │          Input (any of these)               │
                              │   CSV / XLSX / JSON / Parquet / Delta       │
                              └────────────────────┬─────────────────────────┘
                                                   │
                  ┌────────────────────────────────┼─────────────────────────────┐
                  │                                │                             │
                  ▼                                ▼                             ▼
        ┌──────────────────┐              ┌──────────────────┐        ┌──────────────────┐
        │  Local CLI       │              │  Databricks       │        │  Synthetic data  │
        │  python -m       │              │  Notebooks         │        │  generator       │
        │  employee_voice  │              │  databricks_       │        │  eva generate-   │
        │  eva analyze     │              │  notebook.py       │        │  sample          │
        └────────┬─────────┘              └────────┬──────────┘        └──────────────────┘
                 │                                 │
                 │            Privacy layer (Layer 0.5)                  │
                 │  PIIScrubber: regex or Presidio+spaCy                │
                 │  k-anonymity threshold (default 5)                    │
                 └─────────────────────────────────┬─────────────────────┘
                                                   │
        ┌──────────────────────────────────────────┴───────────────────────────────┐
        │                  Medaillon (Bronze / Silver / Gold)                  │
        │                                                                        │
        │  ┌────────────┐   ┌────────────────┐   ┌────────────────────────┐    │
        │  │  Bronze    │ → │    Silver      │ → │        Gold            │    │
        │  │ ingest_    │   │  transform_     │   │     score_gold          │    │
        │  │  bronze()  │   │   silver()      │   │                         │    │
        │  │            │   │                 │   │  + RiskScore, RiskBand │    │
        │  │ raw +      │   │  Layers 0-4:    │   │  + PushFactors,        │    │
        │  │ normalised │   │  preprocess +   │   │    PullFactors         │    │
        │  │ columns    │   │  sentiment +    │   │  + RiskVelocity        │    │
        │  │            │   │  category +     │   │  + k-anonymity         │    │
        │  │            │   │  emotion +      │   │  + optional LLM        │    │
        │  │            │   │  themes         │   │    summary              │    │
        │  │ Persist as │   │  Persist as     │   │  Persist as             │    │
        │  │   bronze   │   │   silver        │   │   gold                  │    │
        │  │   Delta    │   │   Delta         │   │   Delta                 │    │
        │  └────────────┘   └────────────────┘   └────────────────────────┘    │
        │       ↑                                       ↑                       │
        │       └───── re-score without re-running sentiment ─────┘            │
        └──────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼
                                ┌──────────────────────────────┐
                                │  Power BI / Tableau / ML     │
                                └──────────────────────────────┘
```

## Install

```bash
# Minimal local install — CLI + fallback engines + regex PII scrubber.
# No model downloads, runs offline.
pip install employee-voice-analytics[local]

# Production install — adds HuggingFace, BERTopic, Azure OpenAI, Presidio.
pip install employee-voice-analytics[all]

# Databricks / PySpark runtime for the notebook.
pip install employee-voice-analytics[databricks]

# Dev / test toolchain.
pip install employee-voice-analytics[dev]
```

After install both `eva` (Typer + Rich) and `employee-voice` (legacy
argparse) are on `$PATH`.

## Quick start

### Python (recommended)

```python
import pandas as pd
import employee_voice

# In-memory DataFrame -> enriched DataFrame
df = pd.read_csv("data/feedback.csv")
fact = employee_voice.analyze_feedback(
    df,
    redact=True,        # PII-scrub via regex
    k_anonymity=5,       # blank under-threshold slices
    run_topics=True,
)
print(fact[["FeedbackID", "Sentiment", "Category1", "RiskBand",
           "PushFactors", "PullFactors", "RiskVelocity"]].head())

# Medaillon (Bronze / Silver / Gold) — for production Delta pipelines.
# Silver is persisted; Gold is re-scored from Silver without re-running
# the sentiment models. That's the operational win.
stages = employee_voice.analyze_medallion(df, redact=True, k_anonymity=5)
bronze, silver, gold = stages["bronze"], stages["silver"], stages["gold"]
print(f"Bronze: {len(bronze)} rows, contract = {employee_voice.BRONZE_CONTRACT}")
print(f"Silver: {len(silver)} rows, contract = {employee_voice.SILVER_CONTRACT}")
print(f"Gold:   {len(gold)} rows, contract = {employee_voice.GOLD_CONTRACT}")

# Re-score Gold from the same Silver (e.g. after a k-anonymity tweak).
gold_again, _ = employee_voice.score_gold(silver, k_anonymity=5)

# Per-layer building blocks
redacted = employee_voice.scrub(df["Comment"], backend="regex")
aspects  = employee_voice.extract_aspects(["Great manager, awful salary"])
factors  = employee_voice.detect_push_factors(comment)
score, band = employee_voice.score_risk("negative", 0.9, "anger", 0.8,
                                       "Compensation(0.9)", "")

# File in / file out
artifacts = employee_voice.analyze_file(
    "data/feedback.csv", "output/",
    redact=True, k_anonymity=5,
)

# Spark DataFrame
from employee_voice import analyze_spark
enriched_sdf = employee_voice.analyze_spark(spark_df, mlflow_experiment="/Shared/eva")

# Synthetic data for tests / demos
synth = employee_voice.generate_synthetic(n=250, seed=42)
```

See `examples/quickstart.py` for a full walkthrough.

### CLI

```bash
# Generate a realistic synthetic HR dataset
eva generate-sample --rows 250 --out data/sample_feedback_synth.csv

# Run the full pipeline
eva analyze --input data/sample_feedback_synth.csv --outdir output

# With PII redaction + k-anonymity
eva analyze \
    --input data/sample_feedback_synth.csv \
    --outdir output \
    --redact-pii \
    --pii-backend presidio \
    --k-anonymity 5

# Terminal dashboard
eva dashboard --fact output/fact_employee_feedback.csv --by BU
```

The `employee-voice` console script (legacy argparse) is also still
available for backward compatibility.

## Layer enforcement (Layers 0-6)

| # | Layer | Required columns | Runtime deps |
|---|---|---|---|
| 0 | preprocess | `Comment`, `CleanComment` | (none) |
| 0.5 | privacy | `CleanComment` (redacted) | regex: none; presidio: `[privacy]` extra |
| 1 | sentiment | `Sentiment`, `SentimentScore` | `transformers`, `torch` (or fallback) |
| 2 | category | `Category1`, `Category1Score`, `Category2`, `Category2Score`, `AllCategories` | `transformers`, `torch` (or fallback) |
| 3 | emotion | `Emotion`, `EmotionScore` | `transformers`, `torch` (or fallback) |
| 4 | themes | `Topic`, `TopicName`, `TopicKeywords` | `sentence_transformers`, `bertopic`, `umap` (or TF-IDF + KMeans) |
| 5 | risk_score | `RiskScore`, `RiskBand`, `PushFactors`, `PullFactors` | (none) |
| 5b | risk_velocity | `RiskVelocity` | (none; null if `EmployeeID`/`SurveyDate` missing) |
| 6 | llm_summary | (writes side files) | `openai` |

If a layer's dependency is missing, the pipeline raises
`LayerDependencyError` with the matching `pip install` command. To use
the lightweight keyword / TF-IDF fallback engines instead (useful for
CI, dev, or pre-building the BI model before the GPU libraries are
approved), opt in explicitly:

```bash
export EMPLOYEE_VOICE_ALLOW_FALLBACK=1
eva analyze --input data/sample_feedback.csv --outdir output
```

If a layer's runner fails to populate a required column, the pipeline
raises `LayerContractError` instead of silently emitting nulls.

## Privacy guardrails

- **PII scrubber.** `PIIScrubber(backend="regex")` (default) redacts
  emails, phone numbers, SSN, employee IDs, and project codenames
  with no model download. `PIIScrubber(backend="presidio")` uses spaCy
  NER for higher accuracy. Manager context (within 30 chars of
  "manager", "supervisor", etc.) promotes a `PERSON` entity to
  `[MANAGER_NAME]`. Required for any pipeline that sends text to
  external LLM providers.
- **k-anonymity.** Slices (BU × Dept) below the threshold (default 5)
  have their verbatim comments blanked in the fact table; numeric
  score columns are preserved so aggregate dashboards still work.
  Configurable via `--k-anonymity` on `eva analyze`.

## Databricks / PySpark

The `databricks_notebook.py` is a notebook-shaped driver. Two
execution paths, gated by a widget:

| Widget | Default | Behaviour |
|---|---|---|
| `use_spark_udfs` | `true` | `pandas_udf(SCALAR_ITER)` — sentiment / category / emotion / risk run on the executors in parallel. Each executor loads one HF pipeline and reuses it. |
| `use_spark_udfs` | `false` | Driver-side: collect to driver, run `run_all_layers`, write enriched frame back. Simpler to debug; doesn't scale. |
| `mlflow_experiment` | empty | When set, MLflow tracks input row count, output row count, and a pipeline tag. |

### Cluster deployment

1. **Cluster libraries:** `pip install employee-voice-analytics[all,databricks]`
   in the cluster's init script or via a cluster library. The
   `[databricks]` extra pulls PySpark + delta-spark; `[all]` adds the
   HF models (HuggingFace caches them on the cluster on first use).
2. **MLflow:** the `mlflow` package is preinstalled on Databricks
   runtimes; if you're on a custom image, `pip install mlflow` in
   the init script.
3. **Mount the notebook** via Repos or workspace import.
4. **Run All** with the widget defaults, or override per-Job:
   ```
   input_path      dbfs:/mnt/hr/silver/employee_feedback
   output_catalog  hr_prod
   output_database gold
   text_column     Comment
   run_topics      true
   run_summary     false
   use_spark_udfs  true
   mlflow_experiment /Shared/employee-voice-analytics
   ```
5. **Schedule** with a Databricks Job pointing at the notebook.

The Spark runtime writes the same four artifacts as the local CLI
(`fact_employee_feedback`, `dim_topic`, `summary_category`,
`summary_bu`) — Delta tables when `output_catalog` is set, Delta /
Parquet files under `dbfs:/FileStore/employee_voice/output/` otherwise.

## Tests

```bash
pip install 'employee-voice-analytics[dev]'
pytest                       # unit + integration (~17 s, no Java needed)
pytest -m databricks         # also runs the notebook locally (needs Java + PySpark + delta-spark)
pytest -m bench              # smoke benchmark; opt-in for CI
```

## Performance

See [`benchmarks/RESULTS.md`](./benchmarks/RESULTS.md) for the full
report. Headline numbers:

### Medaillon speedup (lexicon backend, single CPU)

| Rows | `analyze_feedback` (full) | `score_gold_from_silver` (Gold only) | **Speedup** |
|---:|---:|---:|---:|
| 1,000 | 6.52 s | 0.10 s | **66×** |
| 10,000 | 10.18 s | 0.88 s | **11.6×** |
| 50,000 | 17.30 s | 4.02 s | **4.3×** |

Re-scoring Gold from a persisted Silver table (after risk-weight or
k-anonymity changes) is the right operational pattern. The benchmark
harness even asserts this as an invariant.

### HF backend (Apple M-series MPS, 100 rows, steady state)

| Layer | rows/sec | Time for 1M rows (single CPU) |
|---|---:|---|
| `sentiment` (RoBERTa) | 22.2 | ~12.5 hours |
| `category` (BART-MNLI zero-shot) | **1.5** | **~7.7 days** |
| `emotion` (GoEmotions) | 24.2 | ~11.5 hours |

**Category is the bottleneck** on a single machine. This is why the
**medaillon split matters for the HF backend**: persisting Silver
and re-scoring Gold from it is the only way the HF backend is
operationally viable. A k-anonymity threshold change takes
milliseconds; a full Silver re-run takes days. For production, run
HF on a Spark cluster via `analyze_spark()` — the `pandas_udf(SCALAR_ITER)`
runtime parallelises per-row inference across executors.

#### Distilled category models

Distilling BART-MNLI to a smaller NLI model gives a 2-5× speedup
on a single machine:

| model | params | rows/sec | vs BART |
|---|---:|---:|---:|
| `facebook/bart-large-mnli` (default) | 400M | 1.5 | 1.0× |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | 140M | 3.8 | 2.5× |
| `cross-encoder/nli-deberta-v3-small` | 140M | 4.1 | 2.7× |
| `typeform/distilbert-base-uncased-mnli` | 70M | 7.9 | **5.3×** |

`typeform/distilbert-base-uncased-mnli` is the practical sweet spot
when single-machine throughput matters more than top-1 accuracy.
Set `CATEGORY_MODEL_ID` in `config.py` to switch. See
[`benchmarks/RESULTS_category_models.md`](./benchmarks/RESULTS_category_models.md)
for the full report including accuracy trade-offs.

```bash
# Lexicon (fast, no model downloads)
python benchmarks/bench.py --rows 1000,10000,50000 --backend lexicon \
    --out benchmarks/RESULTS_lexicon.md --csv benchmarks/results_lexicon.csv

# HuggingFace (slower on CPU, downloads models on first use)
pip install 'employee-voice-analytics[ml]'
python benchmarks/bench_hf.py --rows 100
```

Both harnesses use Python's `subprocess.run` with their own timeout
(no external `timeout` binary needed; works identically on macOS and
Linux).

## Adapting to your company

1. Replace `CATEGORIES` and `HIGH_RISK_CATEGORIES` in `config.py` with
   your own HR taxonomy (1-3 word labels work best for zero-shot).
2. Replace `CATEGORY_KEYWORDS` in `analyzers.py` to match the same
   taxonomy (used by the fallback engine).
3. Extend `PUSH_FACTORS` / `PULL_FACTORS` in `risk_signals.py` with
   your locale / role-specific phrases.
4. Add company-specific intent-to-leave phrases to
   `INTENT_TO_LEAVE_KEYWORDS` in `config.py`.
5. Label ~200 historical comments, measure accuracy of the
   zero-shot categories, then fine-tune with SetFit (20-50 examples
   per category) and swap the model id in `config.py`.
6. Add local-language comments — if volume is material, switch
   sentiment to a multilingual model.

## File layout

```
employee_voice/
    __init__.py
    config.py        taxonomy, model ids, risk weights, layer contracts
    privacy.py       PII scrubber (regex + Presidio) + k-anonymity
    preprocess.py    cleaning, N/A noise filtering
    analyzers.py     sentiment / category / emotion / risk score
    absa.py          aspect-based sentiment (per-aspect polarity)
    risk_signals.py  push/pull factors + q-over-q risk velocity
    topics.py        BERTopic (fallback: TF-IDF + KMeans)
    layers.py        strict layer runner + schema-contract enforcement
    _errors.py       shared exception types
    pipeline.py      orchestration -> fact_employee_feedback
    spark_pipeline.py    PySpark / pandas_udf runtime
    spark_udfs.py        per-executor UDF definitions
    cli.py           legacy argparse CLI (installed as `employee-voice`)
    typer_cli.py     new Typer + Rich CLI (installed as `eva`)
    summarizer.py    optional Azure OpenAI exec summary
    synth_data.py    4-persona synthetic HR feedback generator
    py.typed         PEP 561 marker for type checkers
databricks_notebook.py        notebook driver (Spark + Delta)
data/sample_feedback.csv      20-row hand-written sample
tests/                         pytest suite (129 tests)
    conftest.py
    test_analyzers.py
    test_pipeline.py
    test_layers.py
    test_privacy.py
    test_absa.py
    test_risk_signals.py
    test_spark_pipeline.py
    test_synth_data.py
    test_typer_cli.py
    test_notebook_local.py   marked @pytest.mark.databricks
pyproject.toml                 package metadata, console scripts, extras
requirements.txt              mirrors pyproject.toml core deps
LICENSE, COMMERCIAL_LICENSE.md, README.md
```

## Privacy

Exit-interview and survey comments are sensitive. Keep fact tables in a
restricted workspace, avoid surfacing free-text at individual level in
Power BI (aggregate to team size ≥5), and confirm the data-handling
position before the first production run.
