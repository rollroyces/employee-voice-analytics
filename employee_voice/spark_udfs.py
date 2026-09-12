"""
PySpark UDFs for the Employee Voice Analytics pipeline.

Each UDF uses `pandas_udf(SCALAR_ITER)` so the heavy lifting (HF
pipelines, regex) happens once per executor JVM partition and the
function is called on iterator-yielded pandas Series — which lets
Spark batch small partitions into efficient transformer calls.

Backends:
  - "transformers" (default if available): HuggingFace pipelines.
  - "fallback": the in-package lexicon engines. Selected automatically
    when `transformers` is not importable.
  - "presidio": opt-in for the PII scrubber. Falls back to regex if
    the package or its model is not installed.

These UDFs are registered once per SparkSession via `register_*_udfs()`.
The notebook calls these helpers before invoking `run_spark`.
"""
from __future__ import annotations
import logging
from typing import Iterator

import pandas as pd

log = logging.getLogger(__name__)

# Module-level cache: each executor loads the heavy deps once.
_PIPES: dict = {}
_PII_SCRUBBER = None


def _get_pii_scrubber():
    global _PII_SCRUBBER
    if _PII_SCRUBBER is None:
        from .privacy import PIIScrubber
        # Use presidio if available, else regex.
        try:
            _PII_SCRUBBER = PIIScrubber(backend="presidio")
        except Exception:
            _PII_SCRUBBER = PIIScrubber(backend="regex")
    return _PII_SCRUBBER


def _get_sentiment_pipe():
    from . import config as cfg
    if "sentiment" not in _PIPES:
        from transformers import pipeline
        _PIPES["sentiment"] = pipeline("sentiment-analysis", model=cfg.SENTIMENT_MODEL_ID)
    return _PIPES["sentiment"]


def _get_category_pipe():
    from . import config as cfg
    if "category" not in _PIPES:
        from transformers import pipeline
        _PIPES["category"] = pipeline("zero-shot-classification", model=cfg.CATEGORY_MODEL_ID)
    return _PIPES["category"]


def _get_emotion_pipe():
    from . import config as cfg
    if "emotion" not in _PIPES:
        from transformers import pipeline
        _PIPES["emotion"] = pipeline("text-classification", model=cfg.EMOTION_MODEL_ID)
    return _PIPES["emotion"]


# ---------------------------------------------------------------------------
# PII scrub
# ---------------------------------------------------------------------------
def _scrub_series(texts):
    scrubber = _get_pii_scrubber()
    return [scrubber.scrub(t).text for t in texts]


def register_pii_udf(spark) -> None:
    """Register the PII-scrubbing UDF on the given SparkSession."""
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import StringType

    @pandas_udf(StringType())
    def scrub_pii(s: "pd.Series") -> "pd.Series":
        return s.fillna("").astype(str).map(_scrub_series).map(lambda r: r if isinstance(r, str) else "")

    # Attach to the spark session's namespace for easy import.
    spark.sparkContext._jvm  # ensure JVM is up
    globals()["scrub_pii_udf"] = scrub_pii


# ---------------------------------------------------------------------------
# Sentiment / category / emotion
# ---------------------------------------------------------------------------
def _sentiment_iter(series_iter: "Iterator[pd.Series]") -> "Iterator[pd.DataFrame]":
    """Process sentiment on a stream of pandas Series. Loads the HF
    pipeline once per executor (via the module-level cache) unless
    `ALLOW_FALLBACK` is on, in which case the lexicon engine runs
    unconditionally."""
    pipe = None
    from . import config as _cfg
    if not _cfg.ALLOW_FALLBACK:
        try:
            pipe = _get_sentiment_pipe()
        except Exception:
            pipe = None
    for s in series_iter:
        if pipe is None:
            try:
                pipe = _get_sentiment_pipe()
            except Exception:
                pipe = None
        if pipe is None:
            # Fallback path
            from .analyzers import _fallback_sentiment
            results = [_fallback_sentiment(str(t)) for t in s.fillna("").tolist()]
            yield pd.DataFrame({
                "label": [r[0] for r in results],
                "score": [r[1] for r in results],
            })
        else:
            out = pipe(s.fillna("").tolist(), truncation=True, max_length=512)
            label_map = lambda lbl: (
                "positive" if "pos" in lbl.lower()
                else "negative" if "neg" in lbl.lower()
                else "neutral"
            )
            yield pd.DataFrame({
                "label": [label_map(o["label"]) for o in out],
                "score": [float(o["score"]) for o in out],
            })


def _category_iter(series_iter: "Iterator[pd.Series]") -> "Iterator[pd.DataFrame]":
    from . import config as cfg
    pipe = None
    if not cfg.ALLOW_FALLBACK:
        try:
            pipe = _get_category_pipe()
        except Exception:
            pipe = None
    for s in series_iter:
        if pipe is None:
            try:
                pipe = _get_category_pipe()
            except Exception:
                pipe = None
        if pipe is None:
            from .analyzers import _fallback_category
            rows = []
            for t in s.fillna("").tolist():
                scored = _fallback_category(str(t))
                scored = [(c, s2) for c, s2 in scored if s2 >= cfg.CATEGORY_MIN_SCORE]
                scored.sort(key=lambda x: x[1], reverse=True)
                while len(scored) < cfg.TOP_K_CATEGORIES:
                    scored.append(("", 0.0))
                all_str = " | ".join(f"{c}({s:.2f})" for c, s in scored[:8])
                rows.append({
                    "c1": scored[0][0], "c1s": round(float(scored[0][1]), 4),
                    "c2": scored[1][0], "c2s": round(float(scored[1][1]), 4),
                    "all": all_str,
                })
            yield pd.DataFrame(rows)
        else:
            texts = s.fillna("").tolist()
            out = pipe(texts, candidate_labels=cfg.CATEGORIES, multi_label=True)
            rows = []
            for o in out:
                scored = list(zip(o["labels"], o["scores"]))
                scored = [(c, sc) for c, sc in scored if sc >= cfg.CATEGORY_MIN_SCORE]
                scored.sort(key=lambda x: x[1], reverse=True)
                while len(scored) < cfg.TOP_K_CATEGORIES:
                    scored.append(("", 0.0))
                all_str = " | ".join(f"{c}({sc:.2f})" for c, sc in scored[:8])
                rows.append({
                    "c1": scored[0][0], "c1s": round(float(scored[0][1]), 4),
                    "c2": scored[1][0], "c2s": round(float(scored[1][1]), 4),
                    "all": all_str,
                })
            yield pd.DataFrame(rows)


def _emotion_iter(series_iter: "Iterator[pd.Series]") -> "Iterator[pd.DataFrame]":
    from . import config as _cfg
    pipe = None
    if not _cfg.ALLOW_FALLBACK:
        try:
            pipe = _get_emotion_pipe()
        except Exception:
            pipe = None
    for s in series_iter:
        if pipe is None:
            try:
                pipe = _get_emotion_pipe()
            except Exception:
                pipe = None
        if pipe is None:
            from .analyzers import _fallback_emotion
            rows = []
            for t in s.fillna("").tolist():
                r = _fallback_emotion(str(t))
                rows.append({"label": r[0], "score": r[1]})
            yield pd.DataFrame(rows)
        else:
            out = pipe(s.fillna("").tolist(), truncation=True, max_length=512)
            rows = []
            for r in out:
                if isinstance(r, list):
                    best = max(r, key=lambda x: x["score"])
                else:
                    best = r
                rows.append({"label": best["label"].lower(), "score": float(best["score"])})
            yield pd.DataFrame(rows)


def register_nlp_udfs(spark) -> None:
    """Register sentiment / category / emotion UDFs as SCALAR_ITER
    pandas_udfs. Each gets a per-executor pipeline cache."""
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType

    sent_schema = StructType([StructField("label", StringType()),
                              StructField("score", DoubleType())])
    cat_schema = StructType([
        StructField("c1", StringType()), StructField("c1s", DoubleType()),
        StructField("c2", StringType()), StructField("c2s", DoubleType()),
        StructField("all", StringType()),
    ])
    emo_schema = StructType([StructField("label", StringType()),
                             StructField("score", DoubleType())])

    globals()["analyze_sentiment_udf"] = pandas_udf(_sentiment_iter, sent_schema)
    globals()["analyze_category_udf"] = pandas_udf(_category_iter, cat_schema)
    globals()["analyze_emotion_udf"] = pandas_udf(_emotion_iter, emo_schema)


# ---------------------------------------------------------------------------
# Push / pull factors
# ---------------------------------------------------------------------------
def _push_series(texts):
    from .risk_signals import PUSH_FACTORS, detect_factors
    return [",".join(detect_factors(t, PUSH_FACTORS)) for t in texts]


def _pull_series(texts):
    from .risk_signals import PULL_FACTORS, detect_factors
    return [",".join(detect_factors(t, PULL_FACTORS)) for t in texts]


def register_factor_udfs(spark) -> None:
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import StringType

    globals()["detect_push_udf"] = pandas_udf(lambda s: s.fillna("").astype(str).map(_push_series), StringType())
    globals()["detect_pull_udf"] = pandas_udf(lambda s: s.fillna("").astype(str).map(_pull_series), StringType())


# ---------------------------------------------------------------------------
# Risk score
# ---------------------------------------------------------------------------
def _risk_series(sent, sent_score, cat, all_cats, emo, emo_score, comment):
    """Vectorised risk score. Returns (score, band) Series pair."""
    from .analyzers import score_risk
    scores, bands = [], []
    for i in range(len(sent)):
        s, b = score_risk(
            sentiment=str(sent.iloc[i] or ""),
            sentiment_score=float(sent_score.iloc[i] or 0),
            emotion=str(emo.iloc[i] or ""),
            emotion_score=float(emo_score.iloc[i] or 0),
            all_categories=str(all_cats.iloc[i] or ""),
            comment=str(comment.iloc[i] or ""),
        )
        scores.append(s); bands.append(b)
    return pd.DataFrame({"score": scores, "band": bands})


def register_risk_udf(spark) -> None:
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import StructType, StructField, DoubleType, StringType
    schema = StructType([StructField("score", DoubleType()),
                         StructField("band", StringType())])
    globals()["compute_risk_udf"] = pandas_udf(_risk_series, schema)
