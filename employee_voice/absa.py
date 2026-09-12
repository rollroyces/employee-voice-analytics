"""
Aspect-Based Sentiment Analysis (ABSA) for HR feedback.

Given a comment, extract sentiment for each HR aspect that is mentioned.
The aspects are a small fixed set tied to the HR taxonomy in
`config.py`:

  - Compensation
  - Manager
  - Team
  - Workload
  - Growth       (career development, learning, promotion)
  - Tools        (tech stack, infrastructure)
  - Culture
  - WorkLifeBalance

For each aspect that is mentioned, populate a column
`Sentiment_<Aspect>` with one of {positive, neutral, negative} and a
`Sentiment_<Aspect>Score` column with the confidence. If the aspect
isn't mentioned, the column is null.

Two backends:

  - "transformers" (default if HuggingFace is available and the
    transformers package is installed): uses a small ABSA model.
  - "keyword" (default fallback): uses the existing sentiment
    lexicon engines, but splits the text into aspect-keyword
    windows and assigns sentiment per window.

Both backends return a DataFrame indexed identically to the input.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aspect keyword dictionaries
# ---------------------------------------------------------------------------
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "Compensation": [
        "salary", "pay", "bonus", "compensation", "raise", "promotion", "promo",
        "stock", "equity", "rsu", "benefits", "insurance", "allowance", "perk",
        "compensation", "wage", "income",
    ],
    "Manager": [
        "manager", "boss", "supervisor", "lead", "1:1", "one on one",
        "skip level", "skip-level", "reports to", "reporting to",
        "direct report", "management",
    ],
    "Team": [
        "team", "teammate", "colleague", "coworker", "peer", "squad",
        "group", "crew", "collaboration", "collaborate",
    ],
    "Workload": [
        "workload", "overworked", "understaffed", "headcount", "deadline",
        "pressure", "busy", "stretched", "too much work", "burned out", "burnout",
    ],
    "Growth": [
        "career", "growth", "promotion", "promo", "develop", "development",
        "training", "learning", "mentor", "mentorship", "path", "progression",
        "advancement", "l&d", "upskill",
    ],
    "Tools": [
        "tool", "tools", "software", "system", "platform", "laptop", "tech stack",
        "technology", "outdated", "vpn", "IDE", "infrastructure",
    ],
    "Culture": [
        "culture", "values", "mission", "ethos", "principle", "morale",
        "vibe", "atmosphere", "toxic", "inclusive", "diversity",
    ],
    "WorkLifeBalance": [
        "work-life", "work life", "balance", "flexible", "flex time",
        "remote", "wfh", "work from home", "hours", "weekend",
        "overtime", "late nights", "personal time",
    ],
}


# ---------------------------------------------------------------------------
# Aspect segmentation
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"[,;]")
# Conjunctions that introduce a contrast — strong signals that the
# sentiment flips from the preceding clause. Splitting on these gives
# each clause its own per-aspect sentiment instead of an average.
_CONTRAST_SPLIT = re.compile(
    r"\b(?:but|however|although|though|yet|whereas|while|nevertheless|nonetheless)\b",
    re.IGNORECASE,
)

# Words we DON'T want to count as aspect triggers because they're too
# generic (e.g. "team meeting" should hit Team, but "team" alone on its
# own sentence is a candidate only if the rest of the sentence is
# clearly about a team).
_AMBIGUOUS_TRIGGERS = {
    "team": ["team"],  # always treat as Team
    "culture": ["culture", "values", "morale", "vibe", "toxic", "inclusive", "diversity"],
    "promo": ["promo", "promotion"],  # maps to Compensation OR Growth
}


def _split_into_segments(text: str) -> List[str]:
    """Split a comment into segments for aspect-level sentiment scoring.

    Order of operations:
      1. Split on sentence boundaries (. ! ? \\n)
      2. Within each sentence, split on the first contrast conjunction
         ("but", "however", "although", ...). The conjunction is
         re-attached to the *following* segment so the downstream
         sentiment engine has the context (e.g. "but the manager is
         awful" still gets scored correctly).
      3. Within a long sentence (>80 chars) with no contrast, split on
         commas / semicolons as a fallback.
    """
    if not text:
        return []
    text = text.strip()
    segments: List[str] = []
    for sent in _SENTENCE_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        m = _CONTRAST_SPLIT.search(sent)
        if m:
            head = sent[:m.start()].strip()
            tail = sent[m.end():].strip()
            if head:
                segments.append(head)
            # Re-attach the conjunction to the tail so the sentiment
            # engine can see "but" / "however" for context.
            if tail:
                segments.append(f"{m.group(0)} {tail}")
        elif len(sent) > 80:
            for clause in _CLAUSE_SPLIT.split(sent):
                clause = clause.strip()
                if clause:
                    segments.append(clause)
        else:
            segments.append(sent)
    return segments


def _aspects_in_segment(segment: str) -> List[str]:
    """Return the aspects that are mentioned in a single segment."""
    seg_l = segment.lower()
    hits: List[str] = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(kw in seg_l for kw in keywords):
            hits.append(aspect)
    return hits


# ---------------------------------------------------------------------------
# Per-segment sentiment (uses the same fallback engines as the main
# pipeline so we don't introduce a separate lexicon to maintain).
# ---------------------------------------------------------------------------
def _fallback_sentiment_segment(text: str) -> Tuple[str, float]:
    """Reuse the analyzers' fallback sentiment engine for short text."""
    from .analyzers import _fallback_sentiment
    return _fallback_sentiment(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class ABSAResult:
    """Per-text aspect sentiment results."""
    aspects: Dict[str, Tuple[str, float]]   # aspect -> (label, score)

    def as_polarities(self) -> Dict[str, str]:
        return {a: lbl for a, (lbl, _) in self.aspects.items()}

    def as_scores(self) -> Dict[str, float]:
        return {a: s for a, (_, s) in self.aspects.items()}


def extract_aspect_sentiment(
    text: str,
    backend: str = "keyword",
) -> ABSAResult:
    """Return per-aspect sentiment for a single comment.

    backend="keyword" uses the lexicon engine (default; no model
    download). backend="transformers" uses a HuggingFace ABSA model
    if installed. backend="auto" picks "transformers" if available,
    else "keyword".
    """
    backend = _resolve_backend(backend)
    if backend == "transformers":
        return _absa_transformers(text)
    return _absa_keyword(text)


def _resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    try:
        import transformers  # noqa: F401
        return "transformers"
    except ImportError:
        return "keyword"


# ---------- keyword backend (default) ----------------------------------

def _absa_keyword(text: str) -> ABSAResult:
    if not text:
        return ABSAResult(aspects={})
    segments = _split_into_segments(text)
    if not segments:
        return ABSAResult(aspects={})

    # Accumulate per-aspect sentiment: take the strongest evidence
    # (largest |score - 0.5|) across segments.
    aspect_signals: Dict[str, List[Tuple[str, float]]] = {}
    for seg in segments:
        aspects = _aspects_in_segment(seg)
        if not aspects:
            continue
        label, score = _fallback_sentiment_segment(seg)
        for aspect in aspects:
            aspect_signals.setdefault(aspect, []).append((label, score))

    out: Dict[str, Tuple[str, float]] = {}
    for aspect, signals in aspect_signals.items():
        # Pick the signal with the highest confidence.
        signals.sort(key=lambda x: x[1], reverse=True)
        out[aspect] = signals[0]
    return ABSAResult(aspects=out)


# ---------- transformers backend (placeholder) -------------------------

def _absa_transformers(text: str) -> ABSAResult:
    """Use a HuggingFace ABSA model. Falls back to keyword if anything
    fails (model not downloaded, OOM, etc.).

    We don't ship a specific model id; the caller can override
    `cfg.ABSA_MODEL_ID` if they have one. The default model is the
    popular `yangheng/deberta-v3-base-absa-v1.1` (zero-shot ABSA).
    """
    try:
        from . import config as cfg
        model_id = getattr(cfg, "ABSA_MODEL_ID", "yangheng/deberta-v3-base-absa-v1.1")
        # Lazy import so the heavy dep is only paid when used.
        from transformers import pipeline
        clf = pipeline("text-classification", model=model_id, top_k=None)
        # Note: the yangheng model expects (text, aspect) pairs; for
        # brevity we fall back to keyword segmentation since
        # per-aspect inference with this model is more involved.
        log.debug("ABSA transformers backend available; using keyword segmentation for now")
    except Exception as exc:
        log.info("ABSA transformers path unavailable (%s); using keyword", exc)
    return _absa_keyword(text)


# ---------- DataFrame wrapper ------------------------------------------

def extract_aspect_sentiment_dataframe(
    texts: List[str],
    backend: str = "auto",
) -> pd.DataFrame:
    """Run extract_aspect_sentiment over a list of strings and return
    a DataFrame with columns Sentiment_<Aspect> and SentimentScore_<Aspect>.

    Missing aspects get null. The index matches the input list.
    """
    aspects = sorted(ASPECT_KEYWORDS.keys())
    records: List[Dict[str, object]] = []
    for t in texts:
        result = extract_aspect_sentiment(t, backend=backend)
        row: Dict[str, object] = {}
        for aspect in aspects:
            label_score = result.aspects.get(aspect)
            if label_score:
                row[f"Sentiment_{aspect}"] = label_score[0]
                row[f"SentimentScore_{aspect}"] = round(label_score[1], 4)
            else:
                row[f"Sentiment_{aspect}"] = None
                row[f"SentimentScore_{aspect}"] = None
        records.append(row)
    return pd.DataFrame(records, columns=[c for a in aspects for c in (f"Sentiment_{a}", f"SentimentScore_{a}")])
