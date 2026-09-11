"""
Sentiment / category / emotion analyzers with two backends:

  - "transformers": uses HuggingFace pipelines (slower, higher quality)
  - "fallback":     uses lightweight keyword + lexicon engines (fast, no GPU)

The pipeline always calls analyze_* which auto-detects the available backend.
Output shapes are identical across backends, so swapping is transparent.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import (
    CATEGORIES,
    CATEGORY_MIN_SCORE,
    CATEGORY_MODEL_ID,
    EMOTION_MODEL_ID,
    FORCE_FALLBACK,
    HIGH_RISK_CATEGORIES,
    INTENT_TO_LEAVE_KEYWORDS,
    RISK_BANDS,
    RISK_NEGATIVE_EMOTIONS,
    RISK_WEIGHTS,
    SENTIMENT_MODEL_ID,
    TOP_K_CATEGORIES,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
@dataclass
class _Backends:
    sentiment: str
    category: str
    emotion: str


def _detect_backends() -> _Backends:
    if FORCE_FALLBACK:
        return _Backends("fallback", "fallback", "fallback")
    try:
        import transformers  # noqa: F401
        from transformers import pipeline  # noqa: F401
        return _Backends("transformers", "transformers", "transformers")
    except Exception as exc:  # pragma: no cover
        log.info("transformers unavailable (%s) — using fallback engines", exc)
        return _Backends("fallback", "fallback", "fallback")


_BACKENDS = _detect_backends()
log.info("analyzer backends: %s", _BACKENDS)


# ---------------------------------------------------------------------------
# Lazy-loaded HuggingFace pipelines (only when transformers backend is on)
# ---------------------------------------------------------------------------
_PIPES: Dict[str, object] = {}


def _get_pipe(task: str, model_id: str):
    key = f"{task}::{model_id}"
    if key in _PIPES:
        return _PIPES[key]
    from transformers import pipeline
    pipe = pipeline(task, model=model_id, top_k=None)
    _PIPES[key] = pipe
    return pipe


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------
# Lightweight lexicon fallback sentiment. Three intuitions beyond a bag of words:
# 1. Negation flips polarity within a 3-token window ("not good", "no clear path").
# 2. Phrase-level cues (intent to leave, burnout, "killed my balance") outrank
#    a single positive word earlier in the sentence.
# 3. Magnifiers ("very", "extremely") and intensifiers ("brutal", "constantly")
#    bump confidence rather than just counting tokens.
_NEG_PHRASES = [
    "not kept up", "kept up with the market", "not clear", "no clear",
    "no longer", "no comment", "no communication", "no work-life",
    "no work life", "no recognition", "no mentorship", "no buddy",
    "killing my work-life", "killing my work life", "fed up", "had enough",
    "burned out", "burnt out", "burnout", "below market", "below market pay",
    "out of the loop", "lost the plot", "stalled", "on my way out",
    "layoffs", "redundancy", "demanding and unreasonable", "brutal",
    "non-existent", "non existent", "doesn't work", "doesn't keep",
    "can't keep", "disorganized", "unresponsive", "micromanages",
    "micromanaged", "understaffed", "scripted", "every quarter",
    "kept changing", "keeps changing", "unclear", "frustrated", "frustrating",
    "frustration", "disappointed", "disappointing", "angry", "outraged",
    "exhausted", "exhausting", "stressful", "stress", "anxious", "worried",
    "depressed", "miserable", "unhappy", "toxic", "rude", "hostile",
    "resign", "resigning", "resigned", "leaving", "quit", "interviewing",
    "accepted an offer", "signed an offer", "got an offer",
    "looking for a new job", "looking for another job", "new opportunity",
    "appraisal", "pay cut", "cut the", "budget is a joke",
]
_POS_PHRASES = [
    "great culture", "supportive team", "fair compensation", "happy here",
    "happy with", "really happy", "love the", "loved the", "love working",
    "real growth", "career growth", "promoted last", "mentorship programme",
    "mentorship program", "best", "world class", "world-class",
    "well done", "well done team", "kudos", "thank you", "thank the team",
    "proud", "proud of", "enjoy", "enjoyed", "enjoying",
    "excellent", "amazing", "fantastic", "wonderful", "delighted",
    "thrilled", "supportive", "collaborative", "inclusive",
    "responsive", "clear communication", "well managed", "well organised",
    "well organized", "fair", "balanced", "flexible",
]
_NEG_WORDS = set("""
bad terrible awful horrible hate hated worst worse poor unfair rude toxic
slow unclear broken frustrating frustrated annoying annoyed disappointed
disappointing unhappy dissatisfied exhausting exhausting overworked
underpaid ignored micromanaged micromanages disorganized disorganised
unresponsive brutal demanding unreasonable stagnant stalled obsolete
outdated unfair scripted unclear
""".split())
_POS_WORDS = set("""
good great excellent amazing awesome fantastic love loved loves wonderful
happy satisfied helpful supportive clear fair positive enjoy enjoyed
best better proud collaborative respectful inclusive flexible responsive
delighted thrilled excited grateful thankful
""".split())
_INTENSIFIERS = {"very", "extremely", "really", "so", "constantly", "always", "totally", "completely"}


def _fallback_sentiment(text: str) -> Tuple[str, float]:
    """Lexicon + phrase + negation sentiment. Output one of {positive, neutral, negative}."""
    lower = text.lower()
    pos_score = 0.0
    neg_score = 0.0

    # phrase hits (high signal)
    for phrase in _POS_PHRASES:
        if phrase in lower:
            pos_score += 2.0
    for phrase in _NEG_PHRASES:
        if phrase in lower:
            neg_score += 2.0

    # token-level with simple negation flip in a 3-token look-back window
    tokens = [t.strip(".,!?;:()[]\"'") for t in lower.split()]
    for i, tok in enumerate(tokens):
        window = tokens[max(0, i - 3):i]
        negated = ("not" in window) or ("no" in window)
        if tok in _POS_WORDS:
            if negated:
                neg_score += 1.0
            else:
                pos_score += 1.0
        elif tok in _NEG_WORDS:
            if negated:
                pos_score += 0.5
            else:
                neg_score += 1.0
        elif tok in _INTENSIFIERS:
            # boost whichever side is leading
            if pos_score >= neg_score:
                pos_score += 0.4
            else:
                neg_score += 0.4

    if pos_score == 0 and neg_score == 0:
        return "neutral", 0.55
    if pos_score > neg_score:
        margin = pos_score - neg_score
        conf = min(0.95, 0.55 + 0.07 * margin)
        return "positive", conf
    if neg_score > pos_score:
        margin = neg_score - pos_score
        conf = min(0.95, 0.55 + 0.07 * margin)
        return "negative", conf
    return "neutral", 0.55


def analyze_sentiment(texts: List[str]) -> pd.DataFrame:
    if _BACKENDS.sentiment == "fallback":
        rows = [_fallback_sentiment(t) for t in texts]
        return pd.DataFrame(rows, columns=["Sentiment", "SentimentScore"])
    pipe = _get_pipe("sentiment-analysis", SENTIMENT_MODEL_ID)
    out = pipe(texts, truncation=True, max_length=512)
    rows = []
    for r in out:
        label = r["label"].lower()
        if "pos" in label:
            label = "positive"
        elif "neg" in label:
            label = "negative"
        else:
            label = "neutral"
        rows.append((label, float(r["score"])))
    return pd.DataFrame(rows, columns=["Sentiment", "SentimentScore"])


# ---------------------------------------------------------------------------
# Category (zero-shot) + keyword fallback
# ---------------------------------------------------------------------------
# Per-category keyword hints — used only by the fallback engine.
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Compensation and Benefits": [
        "salary", "pay", "bonus", "compensation", "benefit", "benefits",
        "raise", "promotion", "promo", "stock", "equity", "rsu",
        "insurance", "medical", "dental", "allowance",
    ],
    "Career Development": [
        "career", "promotion", "promo", "growth", "develop", "training",
        "learning", "mentor", "mentorship", "path", "progression",
        "advancement", "next role", "next step",
    ],
    "Leadership": [
        "ceo", "cto", "cfo", "leadership", "executive", "vision",
        "strategy", "direction", "board",
    ],
    "Management": [
        "manager", "management", "supervisor", "boss", "1:1", "one on one",
        "direct report", "reporting line", "skip-level",
    ],
    "Work-Life Balance": [
        "work-life", "work life", "balance", "flexible", "flex time",
        "remote", "wfh", "work from home", "hours", "weekend",
        "overtime", "late nights",
    ],
    "Workload": [
        "workload", "overworked", "too much", "understaffed", "headcount",
        "deadline", "pressure", "busy", "stretched",
    ],
    "Culture and Values": [
        "culture", "values", "mission", "ethos", "belief",
        "principle", "morale",
    ],
    "Team and Collaboration": [
        "team", "teammate", "colleague", "collaboration", "collaborate",
        "silo", "cohesive", "teamwork",
    ],
    "Tools and Technology": [
        "tool", "tools", "software", "system", "platform", "laptop",
        "tech", "technology", "outdated", "slow computer", "vpn",
    ],
    "Physical Workspace": [
        "office", "desk", "workspace", "floor", "pantry", "kitchen",
        "noise", "lighting", "chair", "ergonomic", "wfh setup",
    ],
    "Communication": [
        "communication", "communicate", "transparency", "transparent",
        "update", "announcement", "town hall", "all-hands", "newsletter",
    ],
    "Recognition": [
        "recognition", "acknowledge", "appreciated", "appreciate",
        "credit", "visibility", "reward", "kudos", "thank",
    ],
    "Job Satisfaction": [
        "enjoy", "enjoyed", "fulfilling", "meaningful", "boring",
        "monotonous", "satisfied", "engaged", "engagement",
    ],
    "Diversity and Inclusion": [
        "diversity", "inclusion", "dei", "belonging", "equity",
        "representation", "minority", "underrepresented",
    ],
    "Performance Management": [
        "performance", "review", "kpi", "okr", "goal", "feedback",
        "appraisal", "evaluation",
    ],
    "Onboarding": [
        "onboarding", "first day", "first week", "orientation",
        "ramp", "ramp-up", "buddy",
    ],
    "Company Strategy": [
        "strategy", "roadmap", "direction", "pivot", "vision",
        "reorganization", "restructure", "layoff", "redundancy",
    ],
    "Customer or Stakeholder": [
        "customer", "client", "stakeholder", "user", "patient",
        "tenant", "vendor", "supplier",
    ],
}


def _fallback_category(text: str) -> List[Tuple[str, float]]:
    text_l = text.lower()
    scored: List[Tuple[str, float]] = []
    for cat, kws in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in text_l)
        if hits:
            scored.append((cat, min(0.95, 0.30 + 0.15 * hits)))
    if not scored:
        # weak default — mark as Job Satisfaction with low score
        scored.append(("Job Satisfaction", 0.25))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def analyze_category(texts: List[str]) -> pd.DataFrame:
    """
    Returns columns: Category1, Category1Score, Category2, Category2Score, AllCategories
    AllCategories is a " | cat(score)" string for downstream filtering.
    """
    if _BACKENDS.category == "fallback":
        rows = []
        for t in texts:
            scored = _fallback_category(t)
            rows.append(_pack_categories(scored))
        return pd.DataFrame(rows)

    pipe = _get_pipe("zero-shot-classification", CATEGORY_MODEL_ID)
    rows = []
    # batch through pipeline one-by-one to keep memory bounded
    for t in texts:
        out = pipe(t, candidate_labels=CATEGORIES, multi_label=True)
        scored = list(zip(out["labels"], out["scores"]))
        rows.append(_pack_categories(scored))
    return pd.DataFrame(rows)


def _pack_categories(scored: List[Tuple[str, float]]) -> Dict[str, object]:
    scored = [(c, s) for c, s in scored if s >= CATEGORY_MIN_SCORE]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:TOP_K_CATEGORIES]
    while len(top) < TOP_K_CATEGORIES:
        top.append(("", 0.0))
    all_str = " | ".join(f"{c}({s:.2f})" for c, s in scored[:8])
    return {
        "Category1": top[0][0],
        "Category1Score": round(float(top[0][1]), 4),
        "Category2": top[1][0],
        "Category2Score": round(float(top[1][1]), 4),
        "AllCategories": all_str,
    }


# ---------------------------------------------------------------------------
# Emotion
# ---------------------------------------------------------------------------
# Tiny fallback lexicon mapping emotion words -> label from RISK_NEGATIVE_EMOTIONS
# plus a few positives. Confidence decays with single-word matches.
_EMOTION_LEXICON: Dict[str, List[str]] = {
    "anger": ["angry", "furious", "outraged", "livid", "mad", "rage", "hate"],
    "annoyance": ["annoyed", "irritated", "frustrated", "fed up", "bothered"],
    "disappointment": ["disappointed", "letdown", "expected more", "underwhelmed"],
    "disapproval": ["disapprove", "wrong", "unacceptable", "should not", "shouldn't"],
    "disgust": ["disgusted", "revolted", "sick of"],
    "embarrassment": ["embarrassed", "humiliated", "ashamed"],
    "fear": ["afraid", "scared", "worried", "anxious", "fearful", "uncertain"],
    "grief": ["grief", "mourning", "heartbroken"],
    "remorse": ["sorry", "regret", "regretful", "apologize"],
    "sadness": ["sad", "unhappy", "depressed", "down", "miserable", "crying"],
    "joy": ["happy", "delighted", "thrilled", "excited", "love"],
    "gratitude": ["grateful", "thankful", "appreciate", "thanks"],
    "optimism": ["optimistic", "hopeful", "looking forward"],
    "pride": ["proud", "accomplished"],
    "admiration": ["admire", "respect", "impressive"],
}


def _fallback_emotion(text: str) -> Tuple[str, float]:
    text_l = text.lower()
    best_label, best_hits = "neutral", 0
    for label, kws in _EMOTION_LEXICON.items():
        hits = sum(1 for kw in kws if kw in text_l)
        if hits > best_hits:
            best_label, best_hits = label, hits
    if best_hits == 0:
        return "neutral", 0.55
    return best_label, min(0.95, 0.55 + 0.15 * best_hits)


def analyze_emotion(texts: List[str]) -> pd.DataFrame:
    if _BACKENDS.emotion == "fallback":
        rows = [_fallback_emotion(t) for t in texts]
        return pd.DataFrame(rows, columns=["Emotion", "EmotionScore"])
    pipe = _get_pipe("text-classification", EMOTION_MODEL_ID)
    rows = []
    out = pipe(texts, truncation=True, max_length=512)
    for r in out:
        # top_k=None returns list of {label, score}
        if isinstance(r, list):
            best = max(r, key=lambda x: x["score"])
        else:
            best = r
        rows.append((best["label"].lower(), float(best["score"])))
    return pd.DataFrame(rows, columns=["Emotion", "EmotionScore"])


# ---------------------------------------------------------------------------
# Risk score
# ---------------------------------------------------------------------------
def has_intent_to_leave(text: str) -> bool:
    text_l = text.lower()
    return any(kw in text_l for kw in INTENT_TO_LEAVE_KEYWORDS)


def _matched_high_risk_categories(all_categories: str) -> int:
    if not all_categories:
        return 0
    cats = {c.split("(")[0].strip() for c in all_categories.split(" | ")}
    return sum(1 for c in cats if c in set(HIGH_RISK_CATEGORIES))


def score_risk(
    sentiment: str,
    sentiment_score: float,
    emotion: str,
    emotion_score: float,
    all_categories: str,
    comment: str,
) -> Tuple[float, str]:
    score = 0.0
    if sentiment == "negative":
        score += RISK_WEIGHTS.negative_sentiment * sentiment_score

    n_high_risk = min(
        _matched_high_risk_categories(all_categories),
        RISK_WEIGHTS.max_high_risk_categories,
    )
    score += RISK_WEIGHTS.high_risk_category * n_high_risk

    if emotion in set(RISK_NEGATIVE_EMOTIONS):
        score += RISK_WEIGHTS.negative_emotion * max(emotion_score, 0.5)

    if has_intent_to_leave(comment):
        score += RISK_WEIGHTS.intent_to_leave

    score = min(score, RISK_WEIGHTS.cap)
    band = "Low"
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            band = label
            break
    return round(float(score), 2), band
