"""Tests for the ABSA module."""
from __future__ import annotations
import pytest

from employee_voice.absa import (
    extract_aspect_sentiment,
    extract_aspect_sentiment_dataframe,
    ASPECT_KEYWORDS,
    _split_into_segments,
    _aspects_in_segment,
)


class TestAspectSegmentation:
    def test_split_sentences(self):
        segs = _split_into_segments("Great manager. Bad salary. Team is friendly.")
        assert len(segs) >= 3

    def test_split_long_sentence_on_comma(self):
        # No "but" in this sentence -> falls through to comma split.
        segs = _split_into_segments(
            "My manager is supportive, the salary is below market, "
            "and the workload has been brutal for months."
        )
        assert len(segs) >= 3

    def test_split_on_contrast_conjunction(self):
        # "but" is a contrast cue — should split into two segments.
        segs = _split_into_segments("Loved the team but the manager is awful")
        assert len(segs) == 2
        assert "team" in segs[0].lower()
        assert "manager" in segs[1].lower()

    def test_aspects_in_segment(self):
        aspects = _aspects_in_segment("My manager is great but the salary is below market")
        assert "Manager" in aspects
        assert "Compensation" in aspects

    def test_empty_segment(self):
        assert _aspects_in_segment("") == []
        assert _split_into_segments("") == []


class TestExtractAspectSentiment:
    def test_manager_negative(self):
        r = extract_aspect_sentiment("My manager is unresponsive and never approves time off")
        assert "Manager" in r.aspects
        assert r.aspects["Manager"][0] == "negative"

    def test_compensation_positive(self):
        r = extract_aspect_sentiment("The salary is very competitive and the bonus was generous")
        assert "Compensation" in r.aspects
        assert r.aspects["Compensation"][0] == "positive"

    def test_multiple_aspects(self):
        r = extract_aspect_sentiment(
            "Loved the team but my manager is awful and the salary is below market"
        )
        assert "Team" in r.aspects
        assert "Manager" in r.aspects
        assert "Compensation" in r.aspects
        assert r.aspects["Team"][0] == "positive"
        assert r.aspects["Manager"][0] == "negative"
        assert r.aspects["Compensation"][0] == "negative"

    def test_no_aspects_mentioned(self):
        # "weekend" -> WorkLifeBalance, so a comment containing
        # "weekend" WILL mention an aspect. Use truly aspect-free text.
        r = extract_aspect_sentiment("The parking lot is being repaved soon")
        assert r.aspects == {}

    def test_empty_text(self):
        r = extract_aspect_sentiment("")
        assert r.aspects == {}


class TestDataFrame:
    def test_columns_present(self):
        df = extract_aspect_sentiment_dataframe(
            ["Great manager", "Terrible salary and awful tools"],
            backend="keyword",
        )
        expected_cols = []
        for aspect in ASPECT_KEYWORDS:
            expected_cols.extend([f"Sentiment_{aspect}", f"SentimentScore_{aspect}"])
        for col in expected_cols:
            assert col in df.columns

    def test_row_count(self):
        df = extract_aspect_sentiment_dataframe(["a", "b", "c"])
        assert len(df) == 3

    def test_score_in_range(self):
        df = extract_aspect_sentiment_dataframe(["Great manager and great salary"])
        for col in df.columns:
            if col.startswith("SentimentScore_"):
                vals = df[col].dropna()
                if len(vals):
                    assert (vals >= 0).all() and (vals <= 1).all()

    def test_backend_keyword_explicit(self):
        df = extract_aspect_sentiment_dataframe(
            ["Great manager"], backend="keyword"
        )
        assert df["Sentiment_Manager"].iloc[0] == "positive"
