# %% [markdown]
# # Employee Voice Analytics — Quickstart
#
# This notebook walks through the recommended Python API for the
# `employee-voice-analytics` package. It covers:
#
#   1. Loading + analysing feedback end-to-end (`analyze_feedback`).
#   2. The Bronze / Silver / Gold medaillon split.
#   3. Per-layer building blocks (PII scrub, ABSA, push/pull, risk).
#   4. Spark runtime.
#
# Run it interactively with Jupyter (`jupyter notebook`), or as a
# plain Python file (`python examples/quickstart.py`).

# %%
import os
# Force the lightweight engines so this works offline / in CI. In
# production, install the `[ml]` extra and unset this.
os.environ.setdefault("EMPLOYEE_VOICE_ALLOW_FALLBACK", "1")

import employee_voice
import pandas as pd

print(f"employee-voice-analytics v{employee_voice.__version__}")
print(f"Top-level API: {[n for n in dir(employee_voice) if not n.startswith('_')][:10]}...")

# %% [markdown]
# ## 1. Generate a synthetic HR feedback dataset
#
# `generate_synthetic()` returns a DataFrame with the four canonical
# personas (High Performer Burnout, Exit Interview Dissatisfied, Happy
# Retained, Neutral Tenure) and edge cases (empty / N/A / PII /
# project codenames / emoji / CJK).

# %%
df = employee_voice.generate_synthetic(n=200, seed=42)
print(f"Generated {len(df)} rows across {df['BU'].nunique()} BUs")
print(df[["FeedbackID", "BU", "Comment"]].head(3).to_string())

# %% [markdown]
# ## 2. Run the full pipeline
#
# `analyze_feedback()` takes a DataFrame and returns the enriched
# fact table. PII redaction and k-anonymity are opt-in.

# %%
fact = employee_voice.analyze_feedback(
    df,
    redact=True,        # PII-scrub via regex (default)
    k_anonymity=5,       # blank under-threshold slices
    run_topics=True,
)
print(f"Output rows: {len(fact)} (input: {len(df)})")
print(f"Columns: {len(fact.columns)}")
print()
print("Risk band distribution:")
print(fact["RiskBand"].value_counts().to_string())

# %% [markdown]
# ## 3. Inspect specific signals
#
# The fact table has push factors, pull factors, ABSA signals, and
# risk velocity — pick whichever is most useful for the downstream
# dashboard / ML model.

# %%
print("Top push factors:")
for factor in ["burnout", "manager", "compensation", "career", "workload"]:
    n = sum(factor in s for s in fact["PushFactors"])
    print(f"  {factor:14s} {n:3d} rows")

print()
print("Top pull factors:")
for factor in ["intent_to_leave", "external_offer", "job_search", "competing_pull"]:
    n = sum(factor in s for s in fact["PullFactors"])
    print(f"  {factor:14s} {n:3d} rows")

# %% [markdown]
# ## 4. Bronze / Silver / Gold medaillon
#
# For production Delta pipelines you want each stage persisted
# separately. The Silver table can be re-scored into Gold without
# re-running sentiment — that's the operational win.

# %%
stages = employee_voice.analyze_medallion(df, redact=True, k_anonymity=5)

print("Bronze (raw ingest):")
print(f"  rows: {len(stages['bronze'])}, columns: {len(stages['bronze'].columns)}")
print(f"  contract: {employee_voice.BRONZE_CONTRACT}")

print("\nSilver (Layers 0-4: preprocess + sentiment + category + emotion + themes):")
print(f"  rows: {len(stages['silver'])}, columns: {len(stages['silver'].columns)}")
print(f"  contract: {employee_voice.SILVER_CONTRACT}")

print("\nGold (Layer 5 + privacy: risk + push/pull + velocity + k-anonymity):")
print(f"  rows: {len(stages['gold'])}, columns: {len(stages['gold'].columns)}")
print(f"  contract: {employee_voice.GOLD_CONTRACT}")

# %% [markdown]
# ## 5. Re-score Gold from Silver (no sentiment re-inference)
#
# This is the medaillon payoff: you can iterate on risk weights
# and k-anonymity without paying for sentiment again.

# %%
gold_again, _meta = employee_voice.score_gold(stages["silver"], k_anonymity=5)
print(f"Re-scored gold: {len(gold_again)} rows")
print(f"Risk score delta: {(gold_again['RiskScore'] - stages['gold']['RiskScore']).abs().sum():.4f}")

# %% [markdown]
# ## 6. Per-layer building blocks
#
# You don't have to use `analyze_feedback` for everything. Each
# layer is independently usable for one-off jobs, custom rules,
# and Airflow tasks.

# %%
# PII scrub
sample = "Email me at john.doe@example.com about project ATLAS"
print(f"Original:    {sample}")
print(f"Scrubbed:    {employee_voice.scrub(sample)}")
# Presidio requires the `[privacy]` extra + a spaCy NER model.
# Skip the demo gracefully if the optional dep isn't present so
# the rest of the walkthrough still runs.
try:
    print(f"Presidio:    {employee_voice.scrub(sample, backend='presidio')}")
except ImportError as exc:
    print(f"Presidio:    <skipped — {exc.__class__.__name__}: install [privacy] extra>")
print()

# Aspect-based sentiment
aspects = employee_voice.extract_aspects([
    "Loved the team but my manager is awful and the salary is below market",
])
print("ABSA per aspect:")
for col in aspects.columns:
    val = aspects[col].iloc[0]
    if val is not None and not (isinstance(val, float) and pd.isna(val)):
        print(f"  {col:30s} = {val}")

# Push / pull factors
comment = "Burned out, looking for new job, accepted an offer"
print(f"\nPush factors for '{comment[:40]}...':")
print(f"  {employee_voice.detect_push_factors(comment)}")
print(f"\nPull factors:")
print(f"  {employee_voice.detect_pull_factors(comment)}")

# Single-row risk score
score, band = employee_voice.score_risk(
    sentiment="negative", sentiment_score=0.9,
    emotion="anger", emotion_score=0.85,
    all_categories="Compensation and Benefits(0.9) | Management(0.8)",
    comment=comment,
)
print(f"\nRisk score: {score:.0f} ({band})")

# %% [markdown]
# ## 7. Spark runtime (optional)
#
# If you have PySpark installed (`pip install employee-voice-analytics[databricks]`),
# the same API works on a Spark DataFrame.

# %%
# Comment out the block below if PySpark isn't installed.

# try:
#     from pyspark.sql import SparkSession
#     spark = SparkSession.builder.master("local[1]").getOrCreate()
#     sdf = spark.createDataFrame(df)
#     enriched_sdf = employee_voice.analyze_spark(
#         sdf,
#         redact=True,
#         mlflow_experiment="/Shared/employee-voice",
#     )
#     print(f"Spark enriched rows: {enriched_sdf.count()}")
#     spark.stop()
# except ImportError:
#     print("PySpark not installed — skipping the Spark example.")

# %% [markdown]
# ## 8. CLI equivalent
#
# The same operations are available from the command line via the
# `eva` console script (installed as part of the `[cli]` extra).
#
# ```bash
# # Generate a synthetic dataset
# eva generate-sample --rows 250 --out data/sample_feedback_synth.csv
#
# # Run the pipeline
# eva analyze --input data/sample_feedback_synth.csv --outdir output \
#     --redact-pii --k-anonymity 5
#
# # Render a Rich terminal dashboard
# eva dashboard --fact output/fact_employee_feedback.csv --by BU
# ```
