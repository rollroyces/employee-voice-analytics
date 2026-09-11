# Employee Voice Analytics

Reusable toolkit for sentiment + category + emotion + theme discovery +
attrition-risk scoring on exit interviews, staff surveys and pulse surveys.

A single pipeline. One fact table. Drop the CSVs into Power BI.

## Layout

```
employee_voice/
    config.py        taxonomy, model ids, risk weights   <-- edit this first
    preprocess.py    cleaning, "N/A" noise filtering
    analyzers.py     sentiment / category / emotion / risk score
    topics.py        BERTopic (fallback: TF-IDF + KMeans)
    pipeline.py      orchestration -> fact_employee_feedback
    summarizer.py    optional Azure OpenAI exec summary
run_pipeline.py     CLI runner
databricks_notebook.py   PySpark / Databricks version
tests/test_notebook_local.py   verifies the notebook on a local Spark session
data/sample_feedback.csv
output/             generated
```

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py --input data/sample_feedback.csv --outdir output
```

Your own file:

```bash
python run_pipeline.py --input survey.xlsx --sheet "Responses" --text-column "Q7_Comments"
```

## Databricks / PySpark

```bash
# In a Databricks notebook (or `python databricks_notebook.py`):
#   1. Mount or copy databricks_notebook.py to your workspace.
#   2. Set widgets in the Job UI or leave defaults.
#   3. Run All.
```

Writes Delta tables (`{catalog}.{database}.fact_employee_feedback`,
`dim_topic`, `summary_category`, `summary_bu`) when `output_catalog` is set,
or Delta/Parquet files under `dbfs:/FileStore/employee_voice/output/` otherwise.

To verify the notebook locally before pushing to Databricks:

```bash
export JAVA_HOME=$HOME/.local/jdk/jdk-21.0.12.1.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH
pip install pyspark delta-spark
python tests/test_notebook_local.py
```

The test rewrites `dbfs:/` paths to local and `format("delta")` → `format("parquet")`
on the fly; the production notebook file is unchanged.

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
