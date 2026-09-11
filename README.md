# Employee Voice Analytics

Reusable toolkit for sentiment + category + emotion + theme discovery +
attrition-risk scoring on exit interviews, staff surveys and pulse surveys.

A single pipeline. One fact table. Drop the CSVs into Power BI.

## Layout

```
employee_voice/
    __init__.py
    config.py        taxonomy, model ids, risk weights   <-- edit this first
    preprocess.py    cleaning, "N/A" noise filtering
    analyzers.py     sentiment / category / emotion / risk score
    topics.py        BERTopic (fallback: TF-IDF + KMeans)
    pipeline.py      orchestration -> fact_employee_feedback
    cli.py           argparse CLI (installed as `employee-voice` script)
    summarizer.py    optional Azure OpenAI exec summary
    py.typed         PEP 561 marker for type checkers
databricks_notebook.py   PySpark / Databricks version
run_pipeline.py     thin shim -> employee_voice.cli.main
data/sample_feedback.csv
tests/
    conftest.py
    test_analyzers.py        unit tests (fallback engines)
    test_pipeline.py         integration tests
    test_notebook_local.py   marked @pytest.mark.databricks
pyproject.toml      package metadata, console script, optional deps
requirements.txt    mirrors pyproject.toml core deps
```

## Install

```bash
# Minimal: fallback engines only (no model downloads).
pip install employee-voice-analytics

# With HuggingFace models for Layers 1-4:
pip install 'employee-voice-analytics[ml,topics]'

# Everything including Azure OpenAI + Databricks extras:
pip install 'employee-voice-analytics[all,databricks]'
```

After install the `employee-voice` console script is on `$PATH`.

## Quick start

```bash
# If installed via pip:
employee-voice --input data/sample_feedback.csv --outdir output

# If running from a source clone without installing:
python -m employee_voice --input data/sample_feedback.csv --outdir output
# (legacy: `python run_pipeline.py ...` still works)
```

Your own file:

```bash
employee-voice --input survey.xlsx --sheet "Responses" --text-column "Q7_Comments"
```

## Tests

```bash
pip install 'employee-voice-analytics[dev]'
pytest                       # unit + integration (~3 s, no Java needed)
pytest -m databricks         # also runs the notebook locally (needs Java + pyspark + delta-spark)
```

## Databricks / PySpark

```bash
# In a Databricks notebook (or `python databricks_notebook.py`):
#   1. Attach the wheel: `pip install /path/to/employee_voice_analytics-0.1.0-py3-none-any.whl[databricks]`
#      (or `%pip install -e .` from a source clone).
#   2. Set widgets in the Job UI or leave defaults.
#   3. Run All.
```

Writes Delta tables (`{catalog}.{database}.fact_employee_feedback`,
`dim_topic`, `summary_category`, `summary_bu`) when `output_catalog` is set,
or Delta/Parquet files under `dbfs:/FileStore/employee_voice/output/` otherwise.

## Models

| Layer | Model | Purpose |
|---|---|---|
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Positive / Neutral / Negative |
| Category | `facebook/bart-large-mnli` | zero-shot against the HR taxonomy |
| Emotion | `SamLowe/roberta-base-go_emotions` | 28 emotions |
| Themes | BERTopic + `all-MiniLM-L6-v2` | unsupervised theme discovery |

**Fallback mode:** if `transformers` / `bertopic` are not installed, the toolkit
automatically switches to keyword + TF-IDF/KMeans engines. Output columns are
identical, so you can build the Power BI model first and upgrade later.

## Output — `fact_employee_feedback`

| Column | Notes |
|---|---|
| FeedbackID, SurveyType, SurveyDate, BU, Dept, EmployeeGroup | passed through from source |
| Comment, CleanComment | raw + cleaned |
| Sentiment, SentimentScore | `No Comment` for filtered non-answers |
| Category1/2 + scores, AllCategories | top-2 above `CATEGORY_MIN_SCORE` |
| Emotion, EmotionScore | |
| Topic, TopicName, TopicKeywords | join key to `dim_topic` |
| RiskScore (0–100), RiskBand | High ≥70, Medium ≥40, Low |

Plus `dim_topic.csv`, `summary_category.csv`, `summary_bu.csv`.

## Risk scoring

| Signal | Points |
|---|---|
| Negative sentiment | 40 × confidence |
| High-risk category matched (Compensation, Career, Leadership, Management, WLB) | 20 each, max 2 |
| Negative emotion (anger, disappointment, fear, …) | 20 |
| Intent-to-leave keyword | 30 |

Capped at 100. All weights live in `config.py`.

## Adapting to your company

1. Replace `CATEGORIES` and `HIGH_RISK_CATEGORIES` in `config.py` with your
   own taxonomy (1-3 word labels work best for zero-shot).
2. Replace `CATEGORY_KEYWORDS` in `analyzers.py` to match the same taxonomy
   (used by the fallback engine).
3. Add company-specific intent-to-leave phrases to `INTENT_TO_LEAVE_KEYWORDS`.
4. Label ~200 historical comments, measure accuracy of the zero-shot categories,
   then fine-tune with SetFit (20-50 examples per category) and swap the
   model id in `config.py`.
5. Add local-language comments — if volume is material, switch sentiment
   to a multilingual model (e.g. `cardiffnlp/twitter-xlm-roberta-base-sentiment`).

## Privacy

Exit-interview and survey comments are sensitive. Keep fact tables in a
restricted workspace, avoid surfacing free-text at individual level in Power BI
(aggregate to team size ≥5), and confirm the data-handling position before the
first production run.
