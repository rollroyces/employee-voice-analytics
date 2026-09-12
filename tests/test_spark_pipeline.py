"""Tests for the Spark pipeline (no JVM required for these)."""
from __future__ import annotations
import os

# The spark_pipeline module imports pyspark lazily, but it must be
# importable in a non-Spark context for unit tests and tooling.
import pytest


def test_spark_pipeline_module_imports():
    """The module should be importable even without a live SparkSession."""
    from employee_voice import spark_pipeline  # noqa: F401
    assert hasattr(spark_pipeline, "run_spark")


def test_spark_udfs_module_imports():
    from employee_voice import spark_udfs  # noqa: F401
    assert hasattr(spark_udfs, "register_nlp_udfs")
    assert hasattr(spark_udfs, "register_pii_udf")
    assert hasattr(spark_udfs, "register_factor_udfs")
    assert hasattr(spark_udfs, "register_risk_udf")


def test_run_spark_signature():
    """Spot-check the public API surface so accidental renames are caught."""
    import inspect
    from employee_voice.spark_pipeline import run_spark
    sig = inspect.signature(run_spark)
    expected = {
        "sdf", "text_column", "feedback_id_col", "employee_id_col",
        "survey_date_col", "survey_type_col", "bu_col", "dept_col",
        "employee_group_col", "run_topics", "run_summary",
        "mlflow_experiment", "extra_packages",
    }
    assert expected.issubset(set(sig.parameters.keys()))


def test_udf_iterators_yield_correct_shape():
    """The SCALAR_ITER generators must yield DataFrames with the right
    columns when the lexicon backend is forced."""
    import pandas as pd
    from employee_voice.spark_udfs import _sentiment_iter, _category_iter, _emotion_iter
    # Force fallback engines via env var so we don't load transformers.
    os.environ["EMPLOYEE_VOICE_ALLOW_FALLBACK"] = "1"
    from employee_voice import config
    config.ALLOW_FALLBACK = True
    from employee_voice import analyzers
    analyzers._reset_backends()

    s = pd.Series(["Great team, terrible manager", "Burned out and exhausted"])
    sent_df = next(_sentiment_iter(iter([s])))
    assert list(sent_df.columns) == ["label", "score"]
    assert len(sent_df) == 2

    cat_df = next(_category_iter(iter([s])))
    assert set(cat_df.columns) == {"c1", "c1s", "c2", "c2s", "all"}
    assert len(cat_df) == 2

    emo_df = next(_emotion_iter(iter([s])))
    assert list(emo_df.columns) == ["label", "score"]
    assert len(emo_df) == 2
