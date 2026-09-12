"""
Public Python API for the Employee Voice Analytics toolkit.

This is the recommended entry point for code that imports the package
(`from employee_voice import analyze_feedback`). The CLI commands
(`eva`, `employee-voice`) and the Spark runtime
(`employee_voice.spark_pipeline.run_spark`) both delegate to functions
defined here so the three surfaces stay in lock-step.

Three layers of API:

  - High level:  `analyze_feedback`, `analyze_file`, `analyze_spark`.
                 The simplest way to score a batch of feedback.
  - Per-layer:   `scrub`, `extract_aspects`, `detect_push_factors`,
                 `detect_pull_factors`, `score_risk`,
                 `compute_risk_velocity`. Compose pieces if you only
                 need one signal.
  - Synthetic:   `generate_synthetic` for tests and demos.

All functions are pure with respect to global state: they read
`config.ALLOW_FALLBACK` at call time (so toggles via env vars /
`layers.set_allow_fallback` are honoured), but never mutate it.
"""
from __future__ import annotations
import logging
import os
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from . import __version__
from . import config as cfg
from .layers import run_all_layers

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-exports — the submodules remain importable on their own, but the
# canonical import path is `from employee_voice import <name>`.
# ---------------------------------------------------------------------------
from .absa import (  # noqa: E402,F401
    extract_aspect_sentiment,
    extract_aspect_sentiment_dataframe,
    ASPECT_KEYWORDS,
)
from .risk_signals import (  # noqa: E402,F401
    detect_factors,
    add_factor_columns,
    compute_risk_velocity,
    PUSH_FACTORS,
    PULL_FACTORS,
)
from .privacy import (  # noqa: E402,F401
    PIIScrubber,
    KAnonymityConfig,
    apply_k_anonymity,
)
from .synth_data import generate_synthetic_dataset as _generate_synthetic  # noqa: E402,F401
from .analyzers import score_risk as _score_risk_row  # noqa: E402,F401
from .preprocess import clean_text, is_non_answer  # noqa: E402,F401


# ---------------------------------------------------------------------------
# High level
# ---------------------------------------------------------------------------
def analyze_feedback(
    df: pd.DataFrame,
    *,
    text_column: Optional[str] = None,
    run_topics: bool = True,
    run_summary: bool = False,
    redact: bool = False,
    pii_backend: str = "regex",
    k_anonymity: Optional[int] = None,
    k_anonymity_group_by: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Run the full pipeline on an in-memory DataFrame and return the
    enriched fact table.

    This is the canonical Python entry point. Equivalent to the
    `eva analyze` CLI subcommand, but takes a DataFrame in and returns
    one out — so it composes naturally with pandas, polars, Spark
    `.toPandas()`, Airflow operators, etc.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column with the verbatim feedback text. The
        column is auto-detected (`Comment` / `Comments` / `Feedback` /
        `Response` / `Verbatim` / `Text` / `Answer`); override with
        `text_column=...`. Optional columns that flow through to the
        output: `FeedbackID`, `EmployeeID`, `SurveyDate`, `SurveyType`,
        `BU`, `Dept`, `EmployeeGroup`.
    text_column : str, optional
        Name of the column with the verbatim text. Auto-detected if
        omitted.
    run_topics : bool, default True
        Whether to run Layer 4 (theme discovery). Set False for
        very small corpora where the TF-IDF/KMeans fallback throws.
    run_summary : bool, default False
        Layer 6 (Azure OpenAI exec summary). Requires Azure OpenAI
        creds in env vars; ignored if creds are absent.
    redact : bool, default False
        PII-scrub the comments before analysis. The original Comment
        is preserved; the redacted text is in CleanComment.
    pii_backend : {"regex", "presidio"}, default "regex"
        PII scrubber backend. "presidio" requires the `[privacy]`
        extra.
    k_anonymity : int, optional
        If set, slices with fewer than this many rows (in the
        `k_anonymity_group_by` columns) have their verbatim comments
        blanked in the output. Default None (no suppression).
    k_anonymity_group_by : iterable of str, optional
        Columns to group by for k-anonymity. Default = `("BU", "Dept")`
        when `k_anonymity` is set.

    Returns
    -------
    pd.DataFrame
        The enriched fact table with all layer columns:
        Comment, CleanComment, Sentiment, SentimentScore,
        Category1, Category1Score, Category2, Category2Score,
        AllCategories, Emotion, EmotionScore, Topic, TopicName,
        TopicKeywords, RiskScore, RiskBand, PushFactors, PullFactors,
        RiskVelocity.

    Raises
    ------
    LayerDependencyError
        A layer's runtime dep is missing and `ALLOW_FALLBACK` is off.
    LayerContractError
        A layer's runner failed to populate its required columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"analyze_feedback expects a pandas DataFrame, got {type(df).__name__}")

    work = df.copy()

    # Normalise the text column name first so downstream code can
    # always find "Comment". The auto-detect logic mirrors what the
    # legacy pipeline does — pass `text_column` to override.
    if text_column:
        # The user passed an explicit name. It must exist (case-
        # insensitive), and we raise a "not found" error if it
        # doesn't — different message from the "no column at all"
        # case so callers can tell the two apart.
        match = None
        for c in work.columns:
            if c == text_column or c.lower() == text_column.lower():
                match = c
                break
        if match is None:
            raise ValueError(
                f"--text-column '{text_column}' not found. "
                f"Available: {list(work.columns)}"
            )
        if match != "Comment":
            work = work.rename(columns={match: "Comment"})
    else:
        for cand in ("Comment", "Comments", "Feedback", "Response",
                     "Verbatim", "Text", "Answer"):
            if cand in work.columns:
                if cand != "Comment":
                    work = work.rename(columns={cand: "Comment"})
                break
        else:
            available = list(work.columns)
            raise ValueError(
                "No comment column found. Pass `text_column` explicitly. "
                f"Available: {available}"
            )

    # 1. PII redaction (opt-in). When on, we replace the `Comment`
    #    column with the scrubbed text BEFORE Layer 0 runs, so the
    #    downstream CleanComment is derived from redacted input. The
    #    original verbatim is preserved as `Comment_raw` for audit.
    if redact:
        scrubber = PIIScrubber(backend=pii_backend)
        work["Comment_raw"] = work["Comment"]
        work["Comment"] = scrubber.scrub_series(work["Comment"].fillna("").astype(str))

    # 2. Run the layered pipeline. Pass an outdir so the optional
    #    summary layer has a place to write, but suppress the file
    #    writes below (we return the DataFrame in-memory).
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        layered = run_all_layers(
            work,
            outdir=tmp,
            run_topics=run_topics,
            run_summary_layer=run_summary,
        )
    out = layered["df"]

    # 3. k-anonymity suppression (opt-in). If the user redacted and
    #    applied k-anonymity, the redacted verbatim goes through the
    #    threshold filter — but the *original* Comment_raw is dropped
    #    from the output for privacy (it never lands on disk).
    if redact and "Comment_raw" in out.columns:
        out = out.drop(columns=["Comment_raw"])
    if k_anonymity and k_anonymity > 1:
        groups = list(k_anonymity_group_by or ("BU", "Dept"))
        groups = [g for g in groups if g in out.columns]
        if groups:
            out, _slice_summary = apply_k_anonymity(
                out, groups, KAnonymityConfig(threshold=k_anonymity),
            )

    return out


def analyze_file(
    input_path: str,
    outdir: str,
    *,
    text_column: Optional[str] = None,
    sheet: Optional[str] = None,
    run_topics: bool = True,
    redact: bool = False,
    pii_backend: str = "regex",
    k_anonymity: Optional[int] = None,
    k_anonymity_group_by: Optional[Iterable[str]] = None,
    run_summary: bool = False,
) -> dict:
    """
    Run the full pipeline on a file (CSV / XLSX / JSON / Parquet) and
    write the four standard artifacts to `outdir`.

    Returns
    -------
    dict
        {
            "fact_path":      str,   # path to fact_employee_feedback.csv
            "dim_topic_path": str,   # path to dim_topic.csv
            "summary_path":   str,   # path to summary_category.csv
            "bu_path":        str|None,
            "fact":           pd.DataFrame,  # in-memory fact table
            "topic_meta":     dict,
        }
    """
    import pandas as pd
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(input_path, sheet_name=sheet or 0)
    elif ext == ".json":
        df = pd.read_json(input_path)
    elif ext == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    log.info("Loaded %d rows from %s", len(df), input_path)

    fact = analyze_feedback(
        df,
        text_column=text_column,
        run_topics=run_topics,
        run_summary=run_summary,
        redact=redact,
        pii_backend=pii_backend,
        k_anonymity=k_anonymity,
        k_anonymity_group_by=k_anonymity_group_by,
    )

    os.makedirs(outdir, exist_ok=True)
    fact_path = os.path.join(outdir, cfg.OUTPUT_FACT)
    fact.to_csv(fact_path, index=False)

    # dim_topic / summaries: re-derive from the fact table so they're
    # consistent with whatever redaction / k-anonymity was applied.
    dim_topic_path = os.path.join(outdir, cfg.OUTPUT_DIM_TOPIC)
    if "Topic" in fact.columns and "TopicName" in fact.columns:
        dim = (
            fact.groupby(["Topic", "TopicName", "TopicKeywords"], dropna=False)
            .size()
            .reset_index(name="DocCount")
        )
        dim = dim[["Topic", "TopicName", "TopicKeywords", "DocCount"]]
    else:
        dim = pd.DataFrame(columns=["Topic", "TopicName", "TopicKeywords", "DocCount"])
    dim.to_csv(dim_topic_path, index=False)

    summary_path = os.path.join(outdir, cfg.OUTPUT_SUMMARY_CATEGORY)
    summary = (
        fact.assign(Sentiment=fact["Sentiment"].fillna("No Comment"))
        .pivot_table(index="Category1", columns="Sentiment",
                     values="FeedbackID", aggfunc="count", fill_value=0)
        .reset_index()
    )
    if len(summary.columns) > 1:
        total_cols = [c for c in summary.columns if c != "Category1"]
        if total_cols:
            summary["Total"] = summary[total_cols].sum(axis=1)
            summary = summary.sort_values("Total", ascending=False)
    summary.to_csv(summary_path, index=False)

    bu_path = None
    if "BU" in fact.columns and "RiskScore" in fact.columns:
        bu_path = os.path.join(outdir, cfg.OUTPUT_SUMMARY_BU)
        bu_summary = (
            fact.groupby("BU")
            .agg(
                FeedbackCount=("FeedbackID", "count"),
                AvgRiskScore=("RiskScore", "mean"),
                HighRiskCount=("RiskBand", lambda s: int((s == "High").sum())),
                NegativeCount=("Sentiment", lambda s: int((s == "negative").sum())),
            )
            .reset_index()
            .sort_values("AvgRiskScore", ascending=False)
        )
        bu_summary.to_csv(bu_path, index=False)

    # Re-derive topic_meta for the return value
    topic_meta = {}
    if "Topic" in fact.columns:
        for t in fact["Topic"].dropna().unique():
            sub = fact[fact["Topic"] == t]
            name = sub["TopicName"].iloc[0] if "TopicName" in sub.columns else "outlier"
            kws = (sub["TopicKeywords"].iloc[0] if "TopicKeywords" in sub.columns else "") or ""
            topic_meta[int(t)] = {
                "Name": name,
                "Keywords": [k for k in kws.split(",") if k],
                "Count": int(len(sub)),
            }

    return {
        "fact_path": fact_path,
        "dim_topic_path": dim_topic_path,
        "summary_path": summary_path,
        "bu_path": bu_path,
        "fact": fact,
        "topic_meta": topic_meta,
    }


def analyze_spark(
    sdf,
    *,
    text_column: str = "Comment",
    run_topics: bool = True,
    mlflow_experiment: Optional[str] = None,
    **kwargs,
):
    """
    Run the pipeline on a PySpark DataFrame. Returns a Spark DataFrame
    with the same schema as `analyze_feedback` would produce. See
    `employee_voice.spark_pipeline.run_spark` for the full set of
    keyword arguments.
    """
    from .spark_pipeline import run_spark as _run_spark
    from . import spark_udfs as _udfs
    _udfs.register_pii_udf(sdf.sparkSession)
    _udfs.register_nlp_udfs(sdf.sparkSession)
    _udfs.register_factor_udfs(sdf.sparkSession)
    _udfs.register_risk_udf(sdf.sparkSession)
    return _run_spark(
        sdf,
        text_column=text_column,
        run_topics=run_topics,
        mlflow_experiment=mlflow_experiment,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Per-layer building blocks
# ---------------------------------------------------------------------------
def scrub(text: Union[str, pd.Series], backend: str = "regex") -> Union[str, pd.Series]:
    """
    PII-scrub a single string or a pandas Series of strings. Returns
    the redacted text; pass `backend="presidio"` for higher-accuracy
    NER-based scrubbing (requires the `[privacy]` extra).
    """
    scrubber = PIIScrubber(backend=backend)
    if isinstance(text, pd.Series):
        return scrubber.scrub_series(text)
    return scrubber.scrub(text).text


def extract_aspects(
    texts: Iterable[str],
    backend: str = "keyword",
) -> pd.DataFrame:
    """
    Run aspect-based sentiment analysis over an iterable of strings.
    Returns a DataFrame with `Sentiment_<Aspect>` and
    `SentimentScore_<Aspect>` columns. See `absa.ASPECT_KEYWORDS` for
    the list of aspects.
    """
    return extract_aspect_sentiment_dataframe(list(texts), backend=backend)


def detect_push_factors(text: str) -> list[str]:
    """Return the push factor names (e.g. 'burnout', 'manager') whose
    keywords appear in `text`. Empty list if none."""
    return detect_factors(text, PUSH_FACTORS)


def detect_pull_factors(text: str) -> list[str]:
    """Return the pull factor names (e.g. 'intent_to_leave',
    'job_search') whose keywords appear in `text`."""
    return detect_factors(text, PULL_FACTORS)


def score_risk(
    sentiment: str,
    sentiment_score: float,
    emotion: str,
    emotion_score: float,
    all_categories: str,
    comment: str,
) -> tuple[float, str]:
    """
    Compute the 0-100 RiskScore and the RiskBand ("High" / "Medium" /
    "Low") for a single row. The same function backs Layer 5 of the
    pipeline; this thin wrapper is exposed so external callers (a
    custom rule engine, an Airflow task) can apply the same scoring
    logic to a row that already has the upstream signals.
    """
    return _score_risk_row(
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        emotion=emotion,
        emotion_score=emotion_score,
        all_categories=all_categories,
        comment=comment,
    )


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add PushFactors and PullFactors columns to a DataFrame that
    already has a CleanComment (or Comment) column. Convenience wrapper
    around `risk_signals.add_factor_columns`."""
    return add_factor_columns(df)


def add_risk_velocity(
    df: pd.DataFrame,
    *,
    employee_col: str = "EmployeeID",
    date_col: str = "SurveyDate",
    sentiment_col: str = "Sentiment",
) -> pd.DataFrame:
    """Add a RiskVelocity column (q-over-q change in negative-sentiment
    rate per employee). Convenience wrapper around
    `risk_signals.compute_risk_velocity`."""
    return compute_risk_velocity(
        df,
        employee_col=employee_col,
        date_col=date_col,
        sentiment_col=sentiment_col,
    )


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def generate_synthetic(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic HR feedback dataset of `n` rows across four
    personas (High Performer Burnout, Exit Interview Dissatisfied,
    Happy Retained, Neutral Tenure) with edge cases sprinkled in
    (PII, manager names, project codenames, empty / N/A comments,
    emoji, CJK).
    """
    return _generate_synthetic(n=n, seed=seed)


__all__ = [
    "__version__",
    # High level
    "analyze_feedback",
    "analyze_file",
    "analyze_spark",
    # Per-layer
    "scrub",
    "extract_aspects",
    "detect_push_factors",
    "detect_pull_factors",
    "score_risk",
    "add_factors",
    "add_risk_velocity",
    "extract_aspect_sentiment",
    "extract_aspect_sentiment_dataframe",
    "ASPECT_KEYWORDS",
    "PUSH_FACTORS",
    "PULL_FACTORS",
    "PIIScrubber",
    "KAnonymityConfig",
    "apply_k_anonymity",
    "clean_text",
    "is_non_answer",
    # Synthetic
    "generate_synthetic",
]
