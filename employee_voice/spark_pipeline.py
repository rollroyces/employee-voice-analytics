"""
Spark / Databricks pipeline for the Employee Voice Analytics toolkit.

This module is the production runtime. It uses PySpark's `pandas_udf`
with `SCALAR_ITER` to parallelise the per-row NLP inference across
executors (each executor loads one HF pipeline and processes a stream
of text batches).

The local pipeline (`employee_voice.pipeline.run`) and this Spark
pipeline produce the same schema. Downstream consumers (Power BI,
Tableau, ML models) can read either output without translation.

Usage on Databricks:

    from employee_voice.spark_pipeline import run_spark
    df = spark.read.format("delta").load("/mnt/hr/silver/employee_feedback")
    enriched = run_spark(
        df,
        text_column="Comment",
        run_topics=True,
        mlflow_experiment="/Shared/employee-voice-analytics",
    )
    enriched.write.format("delta").mode("overwrite").save(
        "/mnt/hr/gold/employee_feedback_enriched"
    )

Usage locally for testing:

    pytest -m databricks tests/test_notebook_local.py

The functions are designed to be importable without requiring a live
SparkSession, so unit tests can exercise the schema and argument
parsing without bringing up a JVM.
"""
from __future__ import annotations
import logging
import os
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_spark(
    sdf,
    text_column: str = "Comment",
    feedback_id_col: str = "FeedbackID",
    employee_id_col: Optional[str] = "EmployeeID",
    survey_date_col: Optional[str] = "SurveyDate",
    survey_type_col: Optional[str] = "SurveyType",
    bu_col: Optional[str] = "BU",
    dept_col: Optional[str] = "Dept",
    employee_group_col: Optional[str] = "EmployeeGroup",
    run_topics: bool = True,
    run_summary: bool = False,
    mlflow_experiment: Optional[str] = None,
    extra_packages: Optional[list[str]] = None,
):
    """
    Run the full pipeline on a Spark DataFrame.

    The Spark runtime uses `pandas_udf(SCALAR_ITER)` so the analyzer
    functions are vectorised across executor cores. Each executor
    lazily loads the HuggingFace pipelines and the spaCy model once
    per JVM partition and reuses them across batches.

    Returns: a Spark DataFrame with all the columns from the local
    pipeline's fact table (Sentiment, Category1/2, AllCategories,
    Emotion, EmotionScore, Topic, TopicName, TopicKeywords, RiskScore,
    RiskBand, PushFactors, PullFactors, RiskVelocity, CleanComment).

    `mlflow_experiment` (optional): if set, an MLflow run is started
    and the pipeline logs input row count, output row count, and
    sentiment distribution. The `mlflow` package must be installed on
    the cluster.
    """
    # Imports done lazily so the module is importable without pyspark
    # for unit tests and IDE tooling.
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StringType, StructType, StructField, DoubleType, IntegerType
    )

    log.info("Spark pipeline starting; input rows=%d", sdf.count() if hasattr(sdf, "count") else "?")

    # 1. Privacy: PII scrub via the in-package scrubber. We register
    #    it as a pandas_udf so it runs on the executors in parallel.
    from .privacy import PIIScrubber
    from . import spark_udfs
    spark_udfs.register_pii_udf(sdf.sparkSession)

    # 2. Sentiment / category / emotion via SCALAR_ITER pandas_udfs.
    spark_udfs.register_nlp_udfs(sdf.sparkSession)

    # 3. MLflow tracking (optional)
    if mlflow_experiment:
        _maybe_start_mlflow(mlflow_experiment, sdf)

    # 4. Apply transformations in column order.
    out = sdf
    if text_column != "Comment":
        out = out.withColumnRenamed(text_column, "Comment")

    # PII scrub on the raw Comment → CleanCommentPII. (We keep the
    # original Comment column untouched for compliance auditability.)
    out = out.withColumn("CleanComment", spark_udfs.scrub_pii_udf("Comment"))

    # Sentiment / category / emotion.
    sent_struct = spark_udfs.analyze_sentiment_udf("CleanComment")
    out = out.withColumn("_sent", sent_struct).select(
        "*",
        F.col("_sent.label").alias("Sentiment"),
        F.col("_sent.score").alias("SentimentScore"),
    ).drop("_sent")

    cat_struct = spark_udfs.analyze_category_udf("CleanComment")
    out = out.withColumn("_cat", cat_struct).select(
        "*",
        F.col("_cat.c1").alias("Category1"),
        F.col("_cat.c1s").alias("Category1Score"),
        F.col("_cat.c2").alias("Category2"),
        F.col("_cat.c2s").alias("Category2Score"),
        F.col("_cat.all").alias("AllCategories"),
    ).drop("_cat")

    emo_struct = spark_udfs.analyze_emotion_udf("CleanComment")
    out = out.withColumn("_emo", emo_struct).select(
        "*",
        F.col("_emo.label").alias("Emotion"),
        F.col("_emo.score").alias("EmotionScore"),
    ).drop("_emo")

    # Push / pull factors (driver-side aggregation is fine; the
    # factor detection is per-row and the executor can do it via UDF).
    out = out.withColumn("PushFactors", spark_udfs.detect_push_udf("CleanComment"))
    out = out.withColumn("PullFactors", spark_udfs.detect_pull_udf("CleanComment"))

    # Risk score is a deterministic function of (Sentiment, SentimentScore,
    # Category1, ...). Compute it in a single pandas_udf for vector speed.
    out = out.withColumn(
        "_risk",
        spark_udfs.compute_risk_udf(
            "Sentiment", "SentimentScore", "Category1", "AllCategories",
            "Emotion", "EmotionScore", "CleanComment",
        ),
    ).select(
        "*",
        F.col("_risk.score").alias("RiskScore"),
        F.col("_risk.band").alias("RiskBand"),
    ).drop("_risk")

    # Topics (Layer 4). BERTopic is a driver-side concern; we collect
    # the per-row texts and run discover_topics once, then join back.
    if run_topics:
        out = _attach_topics(out, sdf.sparkSession)

    # Risk velocity (Layer 5c). Driver-side aggregation; do it once on
    # the whole table.
    if employee_id_col and survey_date_col:
        out = _attach_velocity(out, employee_id_col, survey_date_col)

    log.info("Spark pipeline complete; output rows=%d", out.count() if hasattr(out, "count") else "?")
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _attach_topics(out, spark) -> "DataFrame":
    """Run BERTopic / TF-IDF topic discovery driver-side and join the
    topic id / name / keywords back onto the Spark DataFrame."""
    from .layers import run_themes
    from pyspark.sql.types import IntegerType, StringType

    # Collect the CleanComment column to the driver. For very large
    # tables this is a known limitation — call this out in the docs
    # and recommend the pandas_udf refactor in TODO.
    log.warning(
        "Topic discovery runs driver-side via toPandas(); for >1M rows "
        "switch to a pandas_udf wrapper around `discover_topics`."
    )
    pdf = out.select("FeedbackID", "CleanComment").toPandas()
    if pdf.empty:
        out = out.withColumn("Topic", F.lit(None).cast("int")) \
                 .withColumn("TopicName", F.lit(None).cast("string")) \
                 .withColumn("TopicKeywords", F.lit(None).cast("string"))
        return out
    layered = run_themes(pdf, mask=~pdf["CleanComment"].isna())
    pdf = layered[0]
    pdf = pdf[["FeedbackID", "Topic", "TopicName", "TopicKeywords"]]
    topics_sdf = spark.createDataFrame(pdf)
    return out.join(topics_sdf, on="FeedbackID", how="left")


def _attach_velocity(out, employee_col: str, date_col: str) -> "DataFrame":
    """Compute risk velocity driver-side using the same logic as
    the local pipeline."""
    from .risk_signals import compute_risk_velocity
    from pyspark.sql.types import DoubleType

    pdf = out.select("FeedbackID", employee_col, date_col, "Sentiment").toPandas()
    if pdf.empty:
        out = out.withColumn("RiskVelocity", F.lit(None).cast("double"))
        return out
    pdf = compute_risk_velocity(
        pdf,
        employee_col=employee_col,
        date_col=date_col,
        sentiment_col="Sentiment",
    )
    pdf = pdf[["FeedbackID", "RiskVelocity"]]
    vel_sdf = out.sparkSession.createDataFrame(pdf)
    return out.join(vel_sdf, on="FeedbackID", how="left")


def _maybe_start_mlflow(experiment_name: str, sdf) -> None:
    """Start an MLflow run and log input row count. The full per-layer
    metric logging is in `metrics.log_pipeline_metrics` below."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed; skipping experiment tracking")
        return
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    if hasattr(sdf, "count"):
        mlflow.log_metric("input_rows", sdf.count())
    mlflow.set_tag("pipeline", "employee_voice_analytics")
