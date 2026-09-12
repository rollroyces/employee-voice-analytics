"""End-to-end tests for the public Python API (the recommended
import surface — `from employee_voice import analyze_feedback`)."""
from __future__ import annotations
import os

import pandas as pd
import pytest

# Force fallback engines so the test env doesn't need HF models.
os.environ["EMPLOYEE_VOICE_ALLOW_FALLBACK"] = "1"
import employee_voice.config as _cfg  # noqa: E402
_cfg.ALLOW_FALLBACK = True
from employee_voice import analyzers as _analyzers  # noqa: E402
_analyzers._reset_backends()

import employee_voice as eva  # noqa: E402
from employee_voice import (  # noqa: E402
    analyze_feedback, analyze_file, scrub, extract_aspects,
    detect_push_factors, detect_pull_factors, score_risk,
    add_factors, add_risk_velocity, generate_synthetic,
    PIIScrubber, KAnonymityConfig, apply_k_anonymity,
    ASPECT_KEYWORDS, PUSH_FACTORS, PULL_FACTORS, clean_text, is_non_answer,
    __version__,
)


# ---------- smoke / API surface ---------------------------------------

class TestPublicAPI:
    def test_top_level_imports(self):
        for name in ("analyze_feedback", "analyze_file", "scrub",
                     "extract_aspects", "score_risk", "generate_synthetic"):
            assert hasattr(eva, name), f"missing {name!r} on the package"

    def test_version(self):
        assert isinstance(__version__, str) and len(__version__) > 0


# ---------- analyze_feedback ------------------------------------------

class TestAnalyzeFeedback:
    def _df(self):
        return pd.DataFrame({
            "FeedbackID": ["F1", "F2", "F3"],
            "BU": ["Eng", "Eng", "Sales"],
            "Dept": ["A", "A", "X"],
            "EmployeeID": ["E1", "E2", "E3"],
            "SurveyDate": ["2025-01-15", "2025-04-15", "2025-04-15"],
            "Comment": [
                "I love working here, great team and supportive manager",
                "Burned out, looking for new job, accepted an offer",
                "ok",
            ],
        })

    def test_returns_dataframe(self):
        out = analyze_feedback(self._df())
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 3

    def test_layer_columns_present(self):
        out = analyze_feedback(self._df())
        for col in (
            "Comment", "CleanComment", "Sentiment", "SentimentScore",
            "Category1", "Category1Score", "Emotion", "EmotionScore",
            "RiskScore", "RiskBand",
            "PushFactors", "PullFactors", "RiskVelocity",
        ):
            assert col in out.columns, f"missing {col}"

    def test_risk_band_is_categorical(self):
        out = analyze_feedback(self._df())
        assert set(out["RiskBand"].unique()).issubset({"High", "Medium", "Low"})

    def test_risk_score_in_range(self):
        out = analyze_feedback(self._df())
        assert (out["RiskScore"] >= 0).all() and (out["RiskScore"] <= 100).all()

    def test_explicit_text_column(self):
        df = self._df().rename(columns={"Comment": "verbatim"})
        out = analyze_feedback(df, text_column="verbatim")
        assert "Sentiment" in out.columns

    def test_auto_text_column_detection(self):
        # No explicit override; should auto-detect "Comment".
        out = analyze_feedback(self._df())
        assert "Sentiment" in out.columns

    def test_unknown_text_column_raises(self):
        df = self._df().rename(columns={"Comment": "verbatim"})
        with pytest.raises(ValueError, match="not found"):
            analyze_feedback(df, text_column="NopeNopeNope")

    def test_no_comment_column_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="No comment column"):
            analyze_feedback(df)

    def test_type_error_on_non_dataframe(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            analyze_feedback("not a dataframe")  # type: ignore[arg-type]

    def test_run_topics_false(self):
        out = analyze_feedback(self._df(), run_topics=False)
        # Without topics, the Topic column is NaN.
        assert out["Topic"].isna().all()

    def test_redact_true_writes_cleancomment(self):
        out = analyze_feedback(self._df(), redact=True)
        # CleanComment should now contain the redacted text (not the
        # original verbatim, which the scrubber has replaced upstream).
        assert (out["CleanComment"] != "").all()
        # And CleanComment is a string column. (pandas 3.0 infers
        # StringDtype, older versions use object.)
        assert out["CleanComment"].dtype.kind in ("O", "U", "S")

    def test_redact_email(self):
        df = pd.DataFrame({
            "FeedbackID": ["F1"],
            "Comment": ["Email me at john.doe@example.com about project ATLAS"],
        })
        out = analyze_feedback(df, redact=True)
        assert "[EMAIL]" in out["CleanComment"].iloc[0]
        assert "ATLAS" not in out["CleanComment"].iloc[0]

    def test_k_anonymity_blanked_comments(self):
        # 1 row in (Eng, A) — well below threshold of 5.
        df = pd.DataFrame({
            "FeedbackID": ["F1"],
            "BU": ["Eng"],
            "Dept": ["A"],
            "Comment": ["Top secret feedback"],
        })
        out = analyze_feedback(df, k_anonymity=5)
        # The (Eng, A) slice's verbatim column should be blanked.
        assert out["CleanComment"].iloc[0] == ""

    def test_k_anonymity_disabled_by_default(self):
        df = self._df()
        out = analyze_feedback(df)
        # No suppression: all 3 comments should be non-empty.
        assert (out["CleanComment"] != "").sum() == 3

    def test_risk_velocity_populated(self):
        # Same employee across two quarters.
        df = pd.DataFrame({
            "FeedbackID": ["F1", "F2"],
            "EmployeeID": ["E1", "E1"],
            "SurveyDate": ["2025-01-15", "2025-04-15"],
            "Comment": ["Good place to work", "Burned out, leaving"],
        })
        out = analyze_feedback(df)
        # Second row should have a velocity; first row is the baseline.
        assert pd.isna(out["RiskVelocity"].iloc[0]) or out["RiskVelocity"].iloc[0] == 0
        # The Q1 2026 (Q1 2025 in this test) row's velocity should be populated
        # (positive, because sentiment worsened).
        assert out["RiskVelocity"].iloc[1] > 0

    def test_layer_contract_enforced(self):
        # If a layer's runner forgets to populate a column, the
        # pipeline raises rather than silently emitting nulls.
        from employee_voice import layers
        from unittest import mock
        layers.set_allow_fallback(True)
        try:
            with mock.patch.object(layers, "run_sentiment", lambda df, mask: df):
                with pytest.raises((layers.LayerContractError, layers.LayerOrderError)):
                    analyze_feedback(self._df())
        finally:
            layers.set_allow_fallback(False)


# ---------- analyze_file --------------------------------------------

class TestAnalyzeFile:
    def test_runs_on_sample_csv(self, tmp_path):
        # Use the bundled sample.
        sample = os.path.join(os.path.dirname(__file__), "..", "data", "sample_feedback.csv")
        sample = os.path.abspath(sample)
        outdir = tmp_path / "out"
        result = analyze_file(sample, str(outdir))
        for f in ("fact_employee_feedback.csv", "dim_topic.csv",
                  "summary_category.csv", "summary_bu.csv"):
            assert (outdir / f).exists()
        assert isinstance(result["fact"], pd.DataFrame)
        assert len(result["fact"]) > 0
        assert result["bu_path"] is not None  # sample has BU

    def test_returns_paths(self, tmp_path):
        sample = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "data", "sample_feedback.csv"))
        outdir = tmp_path / "out"
        result = analyze_file(sample, str(outdir))
        assert os.path.isfile(result["fact_path"])
        assert os.path.isfile(result["dim_topic_path"])
        assert os.path.isfile(result["summary_path"])
        assert os.path.isfile(result["bu_path"])

    def test_redact_k_anonymity_flags_flow_through(self, tmp_path):
        sample = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "data", "sample_feedback.csv"))
        outdir = tmp_path / "out"
        result = analyze_file(sample, str(outdir), redact=True, k_anonymity=2)
        fact = result["fact"]
        # With k=2 and a small sample, most slices will be above
        # threshold (so their CleanComment is preserved). The 4 files
        # should still all exist regardless of how many were blanked.
        for f in ("fact_employee_feedback.csv", "dim_topic.csv",
                  "summary_category.csv", "summary_bu.csv"):
            assert (outdir / f).exists()


# ---------- per-layer functions -------------------------------------

class TestPerLayer:
    def test_scrub_string(self):
        out = scrub("Email me at john.doe@example.com")
        assert "[EMAIL]" in out

    def test_scrub_series(self):
        s = pd.Series(["john@example.com", "no PII here"])
        out = scrub(s)
        assert "[EMAIL]" in out.iloc[0]
        assert out.iloc[1] == "no PII here"

    def test_scrub_presidio_requires_extra(self):
        # We don't test Presidio here; that's covered in test_privacy.
        # Just ensure the regex path doesn't crash.
        assert "[EMAIL]" in scrub("john@example.com", backend="regex")

    def test_extract_aspects(self):
        df = extract_aspects(["Great manager but terrible salary"])
        # Manager and Compensation should both be present.
        assert "Sentiment_Manager" in df.columns
        assert "Sentiment_Compensation" in df.columns
        assert df["Sentiment_Manager"].iloc[0] == "positive"
        assert df["Sentiment_Compensation"].iloc[0] == "negative"

    def test_detect_push_factors(self):
        out = detect_push_factors("I'm burned out, underpaid, and my manager is unresponsive")
        assert set(out) >= {"burnout", "compensation", "manager"}

    def test_detect_pull_factors(self):
        out = detect_pull_factors("I accepted an offer and I'm interviewing elsewhere")
        assert "external_offer" in out
        assert "job_search" in out

    def test_score_risk_high(self):
        score, band = score_risk(
            sentiment="negative",
            sentiment_score=1.0,
            emotion="anger",
            emotion_score=1.0,
            all_categories="Compensation and Benefits(0.9) | Management(0.9)",
            comment="I am resigning, accepted an offer",
        )
        assert score == 100.0
        assert band == "High"

    def test_score_risk_low(self):
        score, band = score_risk(
            sentiment="positive",
            sentiment_score=0.9,
            emotion="joy",
            emotion_score=0.9,
            all_categories="Team(0.9)",
            comment="Love working here",
        )
        assert band == "Low"

    def test_add_factors(self):
        df = pd.DataFrame({"Comment": ["Burned out, looking for new job"]})
        out = add_factors(df)
        assert "PushFactors" in out.columns
        assert "PullFactors" in out.columns
        assert "burnout" in out["PushFactors"].iloc[0]

    def test_add_risk_velocity(self):
        df = pd.DataFrame({
            "EmployeeID": ["E1", "E1"],
            "SurveyDate": ["2025-01-15", "2025-04-15"],
            "Sentiment": ["positive", "negative"],
        })
        out = add_risk_velocity(df)
        assert "RiskVelocity" in out.columns
        # Q1 2025 has no prior; Q2 (Apr 2025) has velocity vs Q1.
        assert pd.isna(out["RiskVelocity"].iloc[0])


# ---------- synthetic data -------------------------------------------

class TestSyntheticAPI:
    def test_generate_synthetic_basic(self):
        df = generate_synthetic(n=50)
        assert len(df) == 50
        for c in ("FeedbackID", "EmployeeID", "SurveyDate", "BU", "Dept", "Comment"):
            assert c in df.columns

    def test_generate_synthetic_seeded(self):
        a = generate_synthetic(n=50, seed=7)
        b = generate_synthetic(n=50, seed=7)
        assert (a["Comment"].tolist() == b["Comment"].tolist())


# ---------- privacy re-exports ---------------------------------------

class TestPrivacyReExports:
    def test_scrubber_constructable(self):
        s = PIIScrubber(backend="regex")
        # Use a TLD long enough for the email regex; "a@b" is too short.
        assert s.scrub("a@b.com").text == "[EMAIL]"

    def test_k_anonymity_threshold_default(self):
        cfg = KAnonymityConfig()
        assert cfg.threshold == 5

    def test_apply_k_anonymity_known_api(self):
        df = pd.DataFrame({
            "BU": ["Eng"] * 3 + ["Sales"] * 10,
            "Comment": ["x"] * 13,
        })
        out, _summary = apply_k_anonymity(df, ["BU"], KAnonymityConfig(threshold=5))
        # Under-threshold slice blanked, others preserved.
        assert (out[out["BU"] == "Eng"]["Comment"] == "").all()
