"""Unit tests for the analyzer layer (fallback engines). No network required."""
from __future__ import annotations
import os

# Force the lightweight engines so the test never tries to load HF models.
os.environ.setdefault("EMPLOYEE_VOICE_FORCE_FALLBACK", "1")

import pytest

from employee_voice.analyzers import (
    _fallback_sentiment,
    _fallback_category,
    _fallback_emotion,
    analyze_sentiment,
    analyze_category,
    analyze_emotion,
    has_intent_to_leave,
    score_risk,
)


# ---------- Sentiment ---------------------------------------------------

class TestSentiment:
    @pytest.mark.parametrize("text,expected", [
        ("I love working here, the team is amazing", "positive"),
        ("Great culture, supportive manager, fair pay", "positive"),
        ("I hate this place, worst job ever", "negative"),
        ("burned out, fed up, done with it", "negative"),
        ("the meeting is at 3pm", "neutral"),
    ])
    def test_polarity(self, text, expected):
        label, score = _fallback_sentiment(text)
        assert label == expected
        assert 0.0 <= score <= 1.0

    def test_negation_flips_polarity(self):
        # "not good" should not be positive even with positive words nearby
        label, _ = _fallback_sentiment("the manager is not good at all")
        assert label == "negative"

    def test_intent_to_leave_phrases_are_negative(self):
        label, _ = _fallback_sentiment("I've been interviewing and accepted an offer")
        assert label == "negative"

    def test_returns_tuple_for_known_text(self):
        result = analyze_sentiment(["good", "bad"])
        assert list(result.columns) == ["Sentiment", "SentimentScore"]
        assert len(result) == 2

    def test_empty_string(self):
        label, score = _fallback_sentiment("")
        assert label == "neutral"
        assert 0.0 <= score <= 1.0


# ---------- Category ---------------------------------------------------

class TestCategory:
    def test_compensation_keyword_match(self):
        scored = _fallback_category("My salary is below market and bonus is poor")
        labels = [c for c, _ in scored]
        assert "Compensation and Benefits" in labels

    def test_work_life_balance_match(self):
        scored = _fallback_category("Late nights and zero work-life balance")
        labels = [c for c, _ in scored]
        assert "Work-Life Balance" in labels

    def test_unknown_text_falls_back_to_default(self):
        scored = _fallback_category("zxqwerty foobarbaz")
        # Should still return at least one category (the low-confidence default)
        assert len(scored) >= 1

    def test_returns_top_categories(self):
        df = analyze_category(["salary is too low", "career path is unclear"])
        assert "Category1" in df.columns
        assert "Category2" in df.columns
        assert "AllCategories" in df.columns
        assert df["Category1"].iloc[0] == "Compensation and Benefits"
        assert df["Category1"].iloc[1] == "Career Development"


# ---------- Emotion ----------------------------------------------------

class TestEmotion:
    def test_anger_detected(self):
        label, _ = _fallback_emotion("I am angry and outraged")
        assert label == "anger"

    def test_disappointment_detected(self):
        label, _ = _fallback_emotion("I am so disappointed, expected more")
        assert label == "disappointment"

    def test_unknown_returns_neutral(self):
        label, score = _fallback_emotion("asdf qwerty")
        assert label == "neutral"


# ---------- Intent to leave -------------------------------------------

class TestIntentToLeave:
    @pytest.mark.parametrize("text", [
        "I am resigning, last day next month",
        "I've been interviewing and accepted an offer",
        "looking for a new job",
        "I'm out, done with this place",
        "burned out, on my way out",
    ])
    def test_detects(self, text):
        assert has_intent_to_leave(text) is True

    @pytest.mark.parametrize("text", [
        "I love working here",
        "Career growth has been excellent",
    ])
    def test_does_not_detect(self, text):
        assert has_intent_to_leave(text) is False


# ---------- Risk score -------------------------------------------------

class TestRiskScore:
    def test_no_comment_is_low_zero(self):
        score, band = score_risk("neutral", 0.5, "neutral", 0.5, "", "")
        assert score == 0.0
        assert band == "Low"

    def test_high_risk_signal_caps_at_100(self):
        # negative sentiment (40) + 2 high-risk cats (40) + intent-to-leave (30) = 110, capped at 100
        score, band = score_risk(
            sentiment="negative", sentiment_score=1.0,
            emotion="anger", emotion_score=1.0,
            all_categories="Compensation and Benefits(0.9) | Management(0.8)",
            comment="I am resigning, accepted an offer",
        )
        assert score == 100.0
        assert band == "High"

    def test_bands(self):
        # Pure high-risk category match = 20 pts -> still Low band (need >= 40)
        score, band = score_risk("neutral", 0.5, "neutral", 0.5, "Compensation and Benefits(0.9)", "")
        assert band == "Low"
        assert score == 20.0

        # Negative sentiment (40 * 1.0 = 40) + 1 high-risk cat (20) = 60 -> Medium
        score, band = score_risk(
            "negative", 1.0, "neutral", 0.5,
            "Compensation and Benefits(0.9)", "")
        assert 50.0 <= score <= 60.0
        assert band == "Medium"

        # Negative sentiment (40) + 2 high-risk cats (40) + intent-to-leave (30) = 110 -> 100 High
        score, band = score_risk(
            "negative", 1.0, "neutral", 0.5,
            "Compensation and Benefits(0.9) | Management(0.9)",
            "I am resigning")
        assert score == 100.0
        assert band == "High"

    def test_negative_emotion_only(self):
        score, _ = score_risk(
            "neutral", 0.5, "anger", 0.9, "", "")
        # emotion contributes 20 * 0.9 = 18
        assert score == pytest.approx(18.0)
