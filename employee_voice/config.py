"""
Central configuration for the Employee Voice Analytics toolkit.

Edit THIS file first when adapting to a new company:
  - CATEGORIES / HIGH_RISK_CATEGORIES
  - SENTIMENT_MODEL_ID / CATEGORY_MODEL_ID / EMOTION_MODEL_ID / EMBEDDING_MODEL_ID
  - RISK_WEIGHTS / RISK_BANDS
  - NA_PATTERNS / MIN_COMMENT_LEN

Layer enforcement (Layers 0-6):
  The pipeline runs all seven layers in a fixed order. Every layer's
  required output columns are declared in LAYER_CONTRACTS below and
  verified at the end of the run. A missing dep on Layers 1-4 raises
  LayerDependencyError instead of silently using the fallback engine —
  set EMPLOYEE_VOICE_ALLOW_FALLBACK=1 (or call set_allow_fallback(True))
  to opt back into the lightweight engines.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Default category model. For higher throughput on a single machine
# (no GPU cluster), switch to `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
# (smaller, ~10x faster at modest accuracy cost). The HF bench
# (`benchmarks/bench_category_models.py`) measures both.
CATEGORY_MODEL_ID = "facebook/bart-large-mnli"
EMOTION_MODEL_ID = "SamLowe/roberta-base-go_emotions"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"  # for BERTopic

# ---------------------------------------------------------------------------
# Fallback engine opt-in
# ---------------------------------------------------------------------------
# The default behaviour is "real models only": if Layers 1-4 dependencies
# are missing, the pipeline raises LayerDependencyError. Set this True (or
# export EMPLOYEE_VOICE_ALLOW_FALLBACK=1) to permit the keyword + TF-IDF
# fallback engines instead.
_ALLOW_FALLBACK_DEFAULT = os.environ.get("EMPLOYEE_VOICE_ALLOW_FALLBACK", "0") == "1"
ALLOW_FALLBACK: bool = _ALLOW_FALLBACK_DEFAULT


def set_allow_fallback(value: bool) -> None:
    """Toggle the fallback engines. Read by analyzers._detect_backends().

    Also resets the cached backend so the next analyzer call re-detects
    with the new flag value.
    """
    global ALLOW_FALLBACK
    ALLOW_FALLBACK = bool(value)
    # Lazy import to avoid a hard dependency from config -> analyzers
    # (analyzers imports from config at module load).
    from . import analyzers as _analyzers
    _analyzers._reset_backends()


# Zero-shot category candidates. Replace with the deploying company's taxonomy.
# Keep labels short (1-3 words), as zero-shot models perform best on tight labels.
CATEGORIES: List[str] = [
    "Compensation and Benefits",
    "Career Development",
    "Leadership",
    "Management",
    "Work-Life Balance",
    "Workload",
    "Culture and Values",
    "Team and Collaboration",
    "Tools and Technology",
    "Physical Workspace",
    "Communication",
    "Recognition",
    "Job Satisfaction",
    "Diversity and Inclusion",
    "Performance Management",
    "Onboarding",
    "Company Strategy",
    "Customer or Stakeholder",
]

# Categories that count toward attrition risk when matched.
HIGH_RISK_CATEGORIES: List[str] = [
    "Compensation and Benefits",
    "Career Development",
    "Leadership",
    "Management",
    "Work-Life Balance",
]

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# Strings that mean "no answer" and should be filtered before analysis.
NA_PATTERNS: List[str] = [
    "", "n/a", "na", "none", "nil", "no comment", "no comments",
    "nothing", "not applicable", "-", "--", "?", "??", ".",
    "no", "ok", "okay", "good", "fine", "thanks", "thank you",
]
MIN_COMMENT_LEN = 8  # characters after cleaning

# ---------------------------------------------------------------------------
# Risk scoring — points capped at 100.
# ---------------------------------------------------------------------------
@dataclass
class RiskWeights:
    negative_sentiment: float = 40.0   # multiplied by sentiment confidence
    high_risk_category: float = 20.0   # per matched high-risk category, up to 2
    max_high_risk_categories: int = 2
    negative_emotion: float = 20.0
    intent_to_leave: float = 30.0
    cap: float = 100.0

RISK_WEIGHTS = RiskWeights()

# Intent-to-leave keywords — extend per locale / role.
INTENT_TO_LEAVE_KEYWORDS: List[str] = [
    "quit", "leave", "leaving", "resign", "resigning", "resignation",
    "i'm out", "i am out", "on my way out", "last day",
    "looking for another job", "looking for a new job", "new opportunity",
    "interviewing", "offer from", "received an offer",
    "burned out", "burnout", "done with", "fed up", "had enough",
]

# Negative emotions that contribute to risk. Subset of Go-Emotions 28.
RISK_NEGATIVE_EMOTIONS: List[str] = [
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "remorse", "sadness",
]

# Risk bands: (min_score, label). First match wins.
RISK_BANDS: List[Tuple[float, str]] = [
    (70.0, "High"),
    (40.0, "Medium"),
    (0.0, "Low"),
]

# ---------------------------------------------------------------------------
# Output / runtime
# ---------------------------------------------------------------------------
CATEGORY_MIN_SCORE = 0.30   # category must clear this to appear in top-K
TOP_K_CATEGORIES = 2

OUTPUT_FACT = "fact_employee_feedback.csv"
OUTPUT_DIM_TOPIC = "dim_topic.csv"
OUTPUT_SUMMARY_CATEGORY = "summary_category.csv"
OUTPUT_SUMMARY_BU = "summary_bu.csv"

# Folder under output/ for the optional LLM exec summary.
SUMMARY_DIR = "summary"

# Optional Azure OpenAI exec summary — leave None to disable.
AZURE_OPENAI: Dict[str, str | None] = {
    "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
    "deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
}


# ---------------------------------------------------------------------------
# Layer contracts
# ---------------------------------------------------------------------------
# Each layer declares the columns it must populate on the fact DataFrame.
# `run_all_layers()` validates these at the end of a run and raises
# LayerContractError if any required column is missing or entirely null.
#
# Layer 0 (preprocess) and Layer 5 (risk score) are pure-Python and have
# no external deps. Layers 1-4 (sentiment, category, emotion, themes) need
# the HuggingFace / BERTopic stack and require ALLOW_FALLBACK or a working
# backend. Layer 6 (LLM summary) needs the openai package + creds.
@dataclass(frozen=True)
class LayerContract:
    layer: int           # 0..6
    name: str            # human label
    required_columns: Tuple[str, ...]
    depends_on: Tuple[str, ...] = ()  # python module names required at runtime


LAYER_CONTRACTS: Tuple[LayerContract, ...] = (
    LayerContract(0, "preprocess",
                  required_columns=("Comment", "CleanComment"),
                  depends_on=()),
    LayerContract(1, "sentiment",
                  required_columns=("Sentiment", "SentimentScore"),
                  depends_on=("transformers", "torch")),
    LayerContract(2, "category",
                  required_columns=("Category1", "Category1Score",
                                    "Category2", "Category2Score",
                                    "AllCategories"),
                  depends_on=("transformers", "torch")),
    LayerContract(3, "emotion",
                  required_columns=("Emotion", "EmotionScore"),
                  depends_on=("transformers", "torch")),
    LayerContract(4, "themes",
                  required_columns=("Topic", "TopicName", "TopicKeywords"),
                  depends_on=("sentence_transformers", "bertopic", "umap")),
    LayerContract(5, "risk_score",
                  required_columns=("RiskScore", "RiskBand",
                                    "PushFactors", "PullFactors"),
                  depends_on=()),
    LayerContract(6, "llm_summary",
                  required_columns=(),  # optional — writes side files only
                  depends_on=("openai",)),
)
