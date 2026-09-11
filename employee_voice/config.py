"""
Central configuration for the Employee Voice Analytics toolkit.

Edit THIS file first when adapting to a new company:
  - CATEGORIES / HIGH_RISK_CATEGORIES
  - SENTIMENT_MODEL_ID / CATEGORY_MODEL_ID / EMOTION_MODEL_ID / EMBEDDING_MODEL_ID
  - RISK_WEIGHTS / RISK_BANDS
  - NA_PATTERNS / MIN_COMMENT_LEN
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Models — fall back automatically if a package / download is unavailable.
# ---------------------------------------------------------------------------
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
CATEGORY_MODEL_ID = "facebook/bart-large-mnli"
EMOTION_MODEL_ID = "SamLowe/roberta-base-go_emotions"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"  # for BERTopic

# Force the lightweight engines even if transformers/bertopic are installed.
# Useful for CI / CPU-only smoke tests / cost control.
FORCE_FALLBACK = bool(int(os.environ.get("EMPLOYEE_VOICE_FORCE_FALLBACK", "0")))

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
