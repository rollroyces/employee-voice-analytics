"""
Medallion (Bronze / Silver / Gold) data flow for the Employee Voice
Analytics pipeline.

This is the production-shaped version of the layer runner. Instead of
running every layer on every input, the pipeline is split into three
stages that map to the conventional medaillon architecture:

  Bronze:  raw ingest, column normalisation, comment-column
           detection. No NLP. Output: a tidy DataFrame with a
           guaranteed `Comment` column + the original passthrough
           metadata. Persistable as the Bronze Delta table.

  Silver:  Layers 0-4. Preprocess (N/A filter, whitespace cleanup)
           plus sentiment, category, emotion, themes. No risk score,
           no push/pull, no LLM. Output: a per-row fact with all
           `Sentiment_*`, `Category*`, `Emotion*`, `Topic*` columns.
           Persistable as the Silver Delta table.

  Gold:    Layer 5 + privacy. Risk score, push/pull factors, risk
           velocity, k-anonymity suppression, optional LLM summary.
           Output: the final `fact_employee_feedback` ready for
           Power BI / ML. Persistable as the Gold Delta table.

The three stages are independent: you can re-score Gold without
re-running the HF models on the same Silver table. This is the
operational win — sentiment models are the expensive layer; once
you have Silver, Gold is cheap to refresh.

Typical use:

  from employee_voice import (
      ingest_bronze, transform_silver, score_gold,
  )

  bronze = ingest_bronze(df)
  silver = transform_silver(bronze)
  gold   = score_gold(silver, redact=True, k_anonymity=5)
"""
from __future__ import annotations
import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import config as cfg
from .layers import (
    run_preprocess,
    run_sentiment,
    run_category,
    run_emotion,
    run_themes,
    run_risk,
    LayerDependencyError,
    LayerContractError,
)
from .privacy import PIIScrubber, KAnonymityConfig, apply_k_anonymity

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column names that each stage guarantees in its output.
# Used by the tests and by callers that want to assert the stage ran.
# ---------------------------------------------------------------------------
BRONZE_CONTRACT = ("Comment",)
SILVER_CONTRACT = (
    "CleanComment", "Sentiment", "SentimentScore",
    "Category1", "Category1Score", "Category2", "Category2Score",
    "AllCategories", "Emotion", "EmotionScore",
    "Topic", "TopicName", "TopicKeywords",
)
GOLD_CONTRACT = (
    "RiskScore", "RiskBand", "PushFactors", "PullFactors", "RiskVelocity",
)


# ---------------------------------------------------------------------------
# Bronze
# ---------------------------------------------------------------------------
def ingest_bronze(
    df: pd.DataFrame,
    *,
    text_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Bronze stage. Read the input, normalise column names, and
    rename the comment column to `Comment`. No NLP, no redaction
    (redaction is a Silver concern; the verbatim needs to be
    preserved until the privacy filter runs).

    The output is the "raw" view: every original column is kept
    alongside a guaranteed `Comment` column.

    Raises:
        ValueError: if no comment-like column can be found, or the
            user-specified `text_column` doesn't exist.
        TypeError: if the input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"ingest_bronze expects a pandas DataFrame, got {type(df).__name__}")

    work = df.copy()

    # Comment-column detection. Mirrors the api.analyze_feedback logic
    # so the contract is the same regardless of which entry point
    # you use.
    if text_column:
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
            raise ValueError(
                "No comment column found. Pass `text_column` explicitly. "
                f"Available: {list(work.columns)}"
            )

    if "Comment" not in work.columns:
        # Defensive: shouldn't reach here given the checks above.
        raise RuntimeError("Bronze stage produced no Comment column; "
                           "this is a bug in ingest_bronze.")

    return work


# ---------------------------------------------------------------------------
# Silver
# ---------------------------------------------------------------------------
def transform_silver(
    bronze: pd.DataFrame,
    *,
    run_topics: bool = True,
    redact: bool = False,
    pii_backend: str = "regex",
) -> tuple[pd.DataFrame, dict]:
    """
    Silver stage. Run Layers 0-4 (preprocess + sentiment + category
    + emotion + optional themes) on a Bronze DataFrame.

    PII redaction (if `redact=True`) is applied to the `Comment`
    column BEFORE Layer 0, so the rest of the pipeline operates on
    redacted text. The original verbatim is preserved as
    `Comment_raw` for audit (and dropped by the Gold stage's
    privacy filter).

    Returns:
        (silver_df, layered_meta)
        silver_df: a DataFrame containing all Bronze columns plus the
            Silver-contract columns (Sentiment_*, Category_*, Emotion_*,
            Topic_*). `Comment_raw` is included if `redact=True`.
        layered_meta: the dict returned by `run_themes` / `run_risk`
            so callers can introspect topic metadata if needed.
    """
    if not isinstance(bronze, pd.DataFrame):
        raise TypeError(f"transform_silver expects a pandas DataFrame, got {type(bronze).__name__}")

    work = bronze.copy()
    if "Comment" not in work.columns:
        raise ValueError("transform_silver requires a `Comment` column. "
                         "Run ingest_bronze() first.")

    if redact:
        scrubber = PIIScrubber(backend=pii_backend)
        work["Comment_raw"] = work["Comment"]
        work["Comment"] = scrubber.scrub_series(work["Comment"].fillna("").astype(str))

    # Run layers 0-4 directly. We use the layer functions (not
    # run_all_layers) so the contract check at the end doesn't
    # require layer 5 columns. We still get the benefit of the
    # centralized dep probe via run_themes -> discover_topics.
    df, is_non_answer, is_short = run_preprocess(work)
    analyzable_mask = (~is_non_answer) & (~is_short)

    df = run_sentiment(df, analyzable_mask)
    df = run_category(df, analyzable_mask)
    df = run_emotion(df, analyzable_mask)

    topic_meta: dict = {}
    if run_topics:
        df, topic_meta = run_themes(df, analyzable_mask)
    else:
        df["Topic"] = np.nan
        df["TopicName"] = ""
        df["TopicKeywords"] = ""

    return df, {"topic_meta": topic_meta}


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------
def score_gold(
    silver: pd.DataFrame,
    *,
    redact: bool = False,
    pii_backend: str = "regex",
    k_anonymity: Optional[int] = None,
    k_anonymity_group_by: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Gold stage. Run Layer 5 (risk score + push/pull + risk velocity)
    on a Silver DataFrame, apply k-anonymity suppression, and
    optionally scrub any remaining PII.

    The Gold stage is cheap — it operates on the already-scored
    Silver data. This is the layer you re-run when you want to
    tune the risk weights, change the k-anonymity threshold, or
    apply a stricter PII policy without paying for sentiment
    re-inference.

    Returns:
        (gold_df, layered_meta)
        gold_df: the final fact table with all Silver columns plus
            RiskScore, RiskBand, PushFactors, PullFactors,
            RiskVelocity. If `redact=True` and a `Comment_raw`
            column is present, it is dropped from the output.
    """
    if not isinstance(silver, pd.DataFrame):
        raise TypeError(f"score_gold expects a pandas DataFrame, got {type(silver).__name__}")

    work = silver.copy()

    # If we received a Silver table that was produced with
    # redact=True, the original verbatim is in `Comment_raw`. The
    # Gold stage is the right place to drop it.
    if "Comment_raw" in work.columns and not redact:
        # We have a Comment_raw column but the caller didn't ask
        # for redaction at this stage. Drop it so it doesn't
        # accidentally leak into the Gold output.
        work = work.drop(columns=["Comment_raw"])

    if redact:
        # Re-scrub at Gold stage (caller might not have run Silver
        # with redact=True).
        scrubber = PIIScrubber(backend=pii_backend)
        work["Comment_raw"] = work["Comment"]
        work["Comment"] = scrubber.scrub_series(work["Comment"].fillna("").astype(str))

    # Layer 5: risk score + push/pull + risk velocity. The
    # analyzable_mask here is "rows with a non-NoComment Sentiment
    # label", which is what the layer runner expects.
    mask = work.get("Sentiment", pd.Series(["No Comment"] * len(work))) != "No Comment"
    work = run_risk(work, mask)

    if redact and "Comment_raw" in work.columns:
        work = work.drop(columns=["Comment_raw"])

    if k_anonymity and k_anonymity > 1:
        groups = list(k_anonymity_group_by or ("BU", "Dept"))
        groups = [g for g in groups if g in work.columns]
        if groups:
            work, slice_summary = apply_k_anonymity(
                work, groups, KAnonymityConfig(threshold=k_anonymity),
            )
        else:
            slice_summary = pd.DataFrame()
    else:
        slice_summary = pd.DataFrame()

    return work, {"slice_summary": slice_summary}


# ---------------------------------------------------------------------------
# Convenience: full pipeline as one call
# ---------------------------------------------------------------------------
def analyze_medallion(
    df: pd.DataFrame,
    *,
    text_column: Optional[str] = None,
    run_topics: bool = True,
    redact: bool = False,
    pii_backend: str = "regex",
    k_anonymity: Optional[int] = None,
    k_anonymity_group_by: Optional[Iterable[str]] = None,
) -> dict:
    """
    Run the full Bronze -> Silver -> Gold pipeline and return all
    three stages as a dict. Useful when you want to persist each
    stage to its own Delta table.

    Returns:
        dict with keys: "bronze", "silver", "gold" (DataFrames),
        plus "silver_meta" and "gold_meta" for introspection.
    """
    bronze = ingest_bronze(df, text_column=text_column)
    silver, silver_meta = transform_silver(
        bronze, run_topics=run_topics, redact=redact, pii_backend=pii_backend,
    )
    gold, gold_meta = score_gold(
        silver,
        redact=False,  # already redacted in Silver if requested
        pii_backend=pii_backend,
        k_anonymity=k_anonymity,
        k_anonymity_group_by=k_anonymity_group_by,
    )
    return {
        "bronze": bronze,
        "silver": silver,
        "gold": gold,
        "silver_meta": silver_meta,
        "gold_meta": gold_meta,
    }
