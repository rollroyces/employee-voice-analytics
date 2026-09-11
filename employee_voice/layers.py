"""
Layer enforcement for the Employee Voice Analytics pipeline.

`run_all_layers()` is the single entry point the pipeline uses instead of
calling analyzers directly. It:

  1. Checks that every layer's runtime dependency is available, or that
     `ALLOW_FALLBACK` is on. If a dep is missing and fallback is off,
     raises LayerDependencyError with a pip-install hint.
  2. Runs the seven layers in fixed order: preprocess -> sentiment ->
     category -> emotion -> themes -> risk -> (optional) summary.
  3. Validates the per-layer schema contract (LAYER_CONTRACTS in config.py)
     at the end and raises LayerContractError if any required column is
     missing or entirely null.

If a layer legitimately has no data (e.g. no analyzable comments), the
validator allows NaN columns *only if the entire input was non-analyzable*;
otherwise the column must have at least one non-null value.
"""
from __future__ import annotations
import importlib
import logging
from typing import Optional

import numpy as np
import pandas as pd

from . import config as cfg
from ._errors import (
    LayerContractError,
    LayerDependencyError,
    LayerError,
    LayerOrderError,
)
from .preprocess import preprocess_series
from .analyzers import (
    analyze_sentiment,
    analyze_category,
    analyze_emotion,
    score_risk,
)
from .topics import discover_topics

log = logging.getLogger(__name__)


def set_allow_fallback(value: bool) -> None:
    """Convenience re-export of `config.set_allow_fallback`."""
    cfg.set_allow_fallback(value)


# Re-export the error types from this module so existing callers that
# import `from employee_voice.layers import LayerDependencyError` keep
# working.
__all__ = [
    "LayerError",
    "LayerDependencyError",
    "LayerContractError",
    "LayerOrderError",
    "run_all_layers",
    "run_preprocess",
    "run_sentiment",
    "run_category",
    "run_emotion",
    "run_themes",
    "run_risk",
    "run_summary",
]


# ---------------------------------------------------------------------------
# Dependency probe
# ---------------------------------------------------------------------------
def _require_deps(layer: cfg.LayerContract) -> None:
    """Raise LayerDependencyError if any of the layer's `depends_on`
    modules cannot be imported. If ALLOW_FALLBACK is True, missing deps are
    logged at INFO and allowed."""
    missing = [m for m in layer.depends_on if not _module_exists(m)]
    if not missing:
        return
    if cfg.ALLOW_FALLBACK:
        log.info(
            "Layer %d (%s): missing deps %s — falling back to lightweight engines "
            "because ALLOW_FALLBACK is on.",
            layer.layer, layer.name, missing,
        )
        return
    extra = cfg._ALLOW_FALLBACK_DEFAULT  # noqa: SLF001 — internal
    pkg = _extra_name_for_deps(layer.depends_on)
    raise LayerDependencyError(
        f"Layer {layer.layer} ({layer.name}) requires modules {missing} which "
        f"are not importable. Either install them "
        f"(`pip install 'employee-voice-analytics[{pkg}]'`) or set "
        f"EMPLOYEE_VOICE_ALLOW_FALLBACK=1 to use the keyword/TF-IDF fallback."
    )


def _module_exists(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def _extra_name_for_deps(deps: tuple[str, ...]) -> str:
    """Map a tuple of module names to the matching pyproject extras."""
    if "bertopic" in deps or "sentence_transformers" in deps:
        return "ml,topics"
    if "transformers" in deps:
        return "ml"
    if "openai" in deps:
        return "llm"
    return "all"


# ---------------------------------------------------------------------------
# Per-layer runners
# ---------------------------------------------------------------------------
def run_preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Layer 0. Returns (df_with_CleanComment, is_non_answer, is_short)."""
    pp = preprocess_series(df["Comment"])
    df = df.copy()
    df["CleanComment"] = pp["CleanComment"]
    is_non_answer = pp["IsNonAnswer"].astype(bool)
    is_short = pp["IsShort"].astype(bool)
    analyzable_count = int((~is_non_answer & ~is_short).sum())
    log.info("Layer 0 preprocess: %d / %d rows analyzable",
             analyzable_count, len(df))
    return df, is_non_answer, is_short


def run_sentiment(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Layer 1."""
    if mask.any():
        idx = df.index[mask]
        texts = df.loc[idx, "CleanComment"].tolist()
        sent = analyze_sentiment(texts)
        sent.index = idx
        df = pd.concat([df, sent], axis=1)
    else:
        df["Sentiment"] = np.nan
        df["SentimentScore"] = np.nan
    # Non-analyzable rows get a sentinel "No Comment" so Power BI can filter
    # them out without losing the row.
    df["Sentiment"] = df["Sentiment"].astype(object).where(mask, "No Comment")
    df["SentimentScore"] = pd.to_numeric(df["SentimentScore"], errors="coerce").fillna(0.0)
    return df


def run_category(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Layer 2."""
    if mask.any():
        idx = df.index[mask]
        texts = df.loc[idx, "CleanComment"].tolist()
        cat = analyze_category(texts)
        cat.index = idx
        df = pd.concat([df, cat], axis=1)
    else:
        for col in ("Category1", "Category1Score", "Category2",
                    "Category2Score", "AllCategories"):
            df[col] = np.nan
    for col, default in (
        ("Category1", "No Comment"),
        ("Category1Score", 0.0),
        ("Category2", ""),
        ("Category2Score", 0.0),
        ("AllCategories", ""),
    ):
        df[col] = df[col].astype(object).where(mask, default)
    return df


def run_emotion(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Layer 3."""
    if mask.any():
        idx = df.index[mask]
        texts = df.loc[idx, "CleanComment"].tolist()
        emo = analyze_emotion(texts)
        emo.index = idx
        df = pd.concat([df, emo], axis=1)
    else:
        df["Emotion"] = np.nan
        df["EmotionScore"] = np.nan
    df["Emotion"] = df["Emotion"].astype(object).where(mask, "No Comment")
    df["EmotionScore"] = pd.to_numeric(df["EmotionScore"], errors="coerce").fillna(0.0)
    return df


def run_themes(df: pd.DataFrame, mask: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Layer 4. Returns (df, topic_meta)."""
    df["Topic"] = np.nan
    df["TopicName"] = ""
    df["TopicKeywords"] = ""
    meta: dict = {}
    if mask.any():
        idx = df.index[mask]
        texts = df.loc[idx, "CleanComment"].tolist()
        tdf, meta = discover_topics(texts)
        tdf.index = idx
        for col in ("Topic", "TopicName", "TopicKeywords"):
            df.loc[tdf.index, col] = tdf[col].values
    return df, meta


def run_risk(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Layer 5."""
    scores, bands = [], []
    for _, row in df.iterrows():
        if not bool(mask.loc[_]):
            scores.append(0.0); bands.append("Low"); continue
        s, b = score_risk(
            sentiment=str(row.get("Sentiment", "")),
            sentiment_score=float(row.get("SentimentScore") or 0),
            emotion=str(row.get("Emotion") or ""),
            emotion_score=float(row.get("EmotionScore") or 0),
            all_categories=str(row.get("AllCategories") or ""),
            comment=str(row.get("CleanComment") or ""),
        )
        scores.append(s); bands.append(b)
    df["RiskScore"] = scores
    df["RiskBand"] = bands
    return df


def run_summary(df: pd.DataFrame, outdir: str) -> Optional[str]:
    """Layer 6 (optional). Returns path to the exec-summary markdown or None."""
    try:
        from .summarizer import generate_summary
    except ImportError as exc:
        raise LayerDependencyError(
            f"Layer 6 (llm_summary) requires the `openai` package. "
            f"Install with `pip install 'employee-voice-analytics[llm]'`. ({exc})"
        ) from exc
    return generate_summary(df, outdir)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_all_layers(
    df: pd.DataFrame,
    outdir: Optional[str] = None,
    run_topics: bool = True,
    run_summary_layer: bool = False,
) -> dict:
    """Run Layers 0-5 in order, optionally Layer 4 and Layer 6.

    Returns a dict with the augmented DataFrame and per-layer metadata so
    the pipeline can write the topic dimension and exec summary.

    Raises:
        LayerDependencyError — a layer's deps are missing and ALLOW_FALLBACK
            is False.
        LayerContractError — at least one layer's required output columns
            are missing or entirely null after the run.
    """
    by_layer = {c.layer: c for c in cfg.LAYER_CONTRACTS}

    # 1. Probe deps for every layer that will run.
    for c in cfg.LAYER_CONTRACTS:
        if c.layer in (4,) and not run_topics:
            continue
        if c.layer == 6 and not run_summary_layer:
            continue
        _require_deps(c)

    # 2. Layer 0 — preprocess
    df, is_non_answer, is_short = run_preprocess(df)
    analyzable_mask = (~is_non_answer) & (~is_short)

    # 3. Layer 1 — sentiment (must come before risk: score_risk reads sentiment)
    df = run_sentiment(df, analyzable_mask)
    if "Sentiment" not in df.columns:
        raise LayerOrderError("Layer 1 failed to populate 'Sentiment' column")

    # 4. Layer 2 — category
    df = run_category(df, analyzable_mask)
    if "Category1" not in df.columns:
        raise LayerOrderError("Layer 2 failed to populate 'Category1' column")

    # 5. Layer 3 — emotion
    df = run_emotion(df, analyzable_mask)
    if "Emotion" not in df.columns:
        raise LayerOrderError("Layer 3 failed to populate 'Emotion' column")

    # 6. Layer 4 — themes (optional)
    if run_topics:
        df, topic_meta = run_themes(df, analyzable_mask)
    else:
        df["Topic"] = np.nan
        df["TopicName"] = ""
        df["TopicKeywords"] = ""
        topic_meta = {}

    # 7. Layer 5 — risk score
    df = run_risk(df, analyzable_mask)
    if "RiskScore" not in df.columns:
        raise LayerOrderError("Layer 5 failed to populate 'RiskScore' column")

    # 8. Layer 6 — optional LLM summary
    summary_path: Optional[str] = None
    if run_summary_layer and outdir is not None:
        summary_path = run_summary(df, outdir)

    # 9. Validate the contract for every layer that ran.
    contract_failures: list[str] = []
    for c in cfg.LAYER_CONTRACTS:
        if c.layer in (4,) and not run_topics:
            continue
        if c.layer == 6 and not run_summary_layer:
            continue
        # Layer 0 columns are always present (preprocess owns them).
        if c.layer == 0:
            if "Comment" not in df.columns or "CleanComment" not in df.columns:
                contract_failures.append(
                    f"Layer 0 ({c.name}): missing required column(s) {c.required_columns}"
                )
            continue
        # Optional Layer 6 doesn't write to the fact table.
        if not c.required_columns:
            continue
        for col in c.required_columns:
            if col not in df.columns:
                contract_failures.append(
                    f"Layer {c.layer} ({c.name}): required column '{col}' is missing"
                )
                continue
            series = df[col]
            # The contract allows NaN *only* if the entire input was
            # non-analyzable (so every row is a 'No Comment' placeholder).
            non_null = int(series.notna().sum())
            if non_null == 0 and analyzable_mask.any():
                contract_failures.append(
                    f"Layer {c.layer} ({c.name}): required column '{col}' is "
                    f"entirely null on {int(analyzable_mask.sum())} analyzable rows"
                )

    if contract_failures:
        raise LayerContractError(
            "Layer contract violations:\n  - " + "\n  - ".join(contract_failures)
        )

    return {
        "df": df,
        "topic_meta": topic_meta,
        "summary_path": summary_path,
        "analyzable_mask": analyzable_mask,
    }
