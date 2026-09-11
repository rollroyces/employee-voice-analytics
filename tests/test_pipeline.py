"""Integration tests: full pipeline against the bundled sample data."""
from __future__ import annotations
import os
from pathlib import Path

import pandas as pd
import pytest

# Force the fallback engines — the HF backend would download GB of weights.
os.environ.setdefault("EMPLOYEE_VOICE_ALLOW_FALLBACK", "1")

# Toggle the global before any analyzer module touches it.
from employee_voice import config as _cfg  # noqa: E402
_cfg.ALLOW_FALLBACK = True

from employee_voice.pipeline import run  # noqa: E402
from employee_voice.preprocess import clean_text, is_non_answer  # noqa: E402
from employee_voice import layers  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "sample_feedback.csv"


# ---------- Preprocess --------------------------------------------------

class TestPreprocess:
    def test_clean_lowercases_and_strips(self):
        assert clean_text("  Hello   WORLD  ") == "Hello WORLD"

    def test_clean_strips_urls(self):
        assert "http" not in clean_text("see https://example.com for details")

    @pytest.mark.parametrize("text", ["", "n/a", "no comment", ".", "-", "OK"])
    def test_non_answers(self, text):
        assert is_non_answer(text) is True

    @pytest.mark.parametrize("text", [
        "Manager was great but salary below market",
        "Burned out, looking for a new opportunity",
    ])
    def test_answers(self, text):
        assert is_non_answer(text) is False


# ---------- Pipeline ----------------------------------------------------

class TestPipeline:
    def test_runs_on_sample(self, tmp_path):
        artifacts = run(
            input_path=str(DATA),
            outdir=str(tmp_path),
        )
        assert (tmp_path / "fact_employee_feedback.csv").exists()
        assert (tmp_path / "dim_topic.csv").exists()
        assert (tmp_path / "summary_category.csv").exists()
        assert (tmp_path / "summary_bu.csv").exists()

    def test_fact_schema(self, tmp_path):
        artifacts = run(input_path=str(DATA), outdir=str(tmp_path))
        fact = artifacts["fact"]
        required = {
            "FeedbackID", "SurveyType", "BU", "Comment", "CleanComment",
            "Sentiment", "SentimentScore",
            "Category1", "Category1Score", "Category2", "AllCategories",
            "Emotion", "EmotionScore",
            "Topic", "TopicName", "TopicKeywords",
            "RiskScore", "RiskBand",
        }
        assert required.issubset(set(fact.columns))

    def test_no_comment_rows_preserved(self, tmp_path):
        artifacts = run(input_path=str(DATA), outdir=str(tmp_path))
        fact = artifacts["fact"]
        no_comment = fact[fact["Sentiment"] == "No Comment"]
        assert len(no_comment) >= 1  # sample contains at least one
        assert (no_comment["RiskScore"] == 0.0).all()

    def test_risk_band_is_categorical(self, tmp_path):
        artifacts = run(input_path=str(DATA), outdir=str(tmp_path))
        fact = artifacts["fact"]
        assert set(fact["RiskBand"].unique()).issubset({"High", "Medium", "Low"})

    def test_risk_score_in_range(self, tmp_path):
        artifacts = run(input_path=str(DATA), outdir=str(tmp_path))
        fact = artifacts["fact"]
        scores = fact["RiskScore"].astype(float)
        assert (scores >= 0).all() and (scores <= 100).all()

    def test_no_topics_flag(self, tmp_path):
        artifacts = run(input_path=str(DATA), outdir=str(tmp_path), run_topics=False)
        # Without topics, the Topic column should be NaN
        assert artifacts["fact"]["Topic"].isna().all()
        # And dim_topic should be empty (only header)
        dim = pd.read_csv(tmp_path / "dim_topic.csv")
        assert len(dim) == 0

    def test_text_column_override(self, tmp_path):
        # Sample's column is named "Comment" but we can rename via override.
        artifacts = run(
            input_path=str(DATA),
            outdir=str(tmp_path),
            text_column="Comment",
        )
        assert len(artifacts["fact"]) == 20

    def test_unknown_text_column_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            run(input_path=str(DATA), outdir=str(tmp_path), text_column="NopeNopeNope")
