"""Tests for the Bronze / Silver / Gold medaillon split."""
from __future__ import annotations
import os

import pandas as pd
import pytest

# Force fallback engines.
os.environ.setdefault("EMPLOYEE_VOICE_ALLOW_FALLBACK", "1")
import employee_voice.config as _cfg  # noqa: E402
_cfg.ALLOW_FALLBACK = True
from employee_voice import analyzers as _analyzers  # noqa: E402
_analyzers._reset_backends()

import employee_voice as eva  # noqa: E402
from employee_voice.medallion import (  # noqa: E402
    ingest_bronze, transform_silver, score_gold, analyze_medallion,
    BRONZE_CONTRACT, SILVER_CONTRACT, GOLD_CONTRACT,
)


# ---------- Bronze -----------------------------------------------------

class TestBronze:
    def _df(self):
        return pd.DataFrame({
            "FeedbackID": ["F1", "F2"],
            "BU": ["Eng", "Sales"],
            "Comment": ["Love the team", "Burned out"],
        })

    def test_guarantees_comment_column(self):
        out = ingest_bronze(self._df())
        assert "Comment" in out.columns

    def test_passes_through_original_columns(self):
        out = ingest_bronze(self._df())
        for c in ("FeedbackID", "BU"):
            assert c in out.columns

    def test_renames_alternate_text_column(self):
        df = self._df().rename(columns={"Comment": "verbatim"})
        out = ingest_bronze(df, text_column="verbatim")
        assert "Comment" in out.columns

    def test_auto_detect(self):
        out = ingest_bronze(self._df())
        # The "Comment" column was already named correctly.
        assert out["Comment"].iloc[0] == "Love the team"

    def test_unknown_text_column_raises(self):
        df = self._df().rename(columns={"Comment": "verbatim"})
        with pytest.raises(ValueError, match="not found"):
            ingest_bronze(df, text_column="NopeNopeNope")

    def test_no_comment_column_raises(self):
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="No comment column"):
            ingest_bronze(df)

    def test_type_error(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            ingest_bronze("not a df")  # type: ignore[arg-type]


# ---------- Silver -----------------------------------------------------

class TestSilver:
    def _bronze(self):
        return pd.DataFrame({
            "FeedbackID": ["F1", "F2", "F3"],
            "BU": ["Eng", "Eng", "Sales"],
            "EmployeeID": ["E1", "E1", "E2"],
            "SurveyDate": ["2025-01-15", "2025-04-15", "2025-04-15"],
            "Comment": [
                "Great team, supportive manager",
                "Burned out, looking for new job",
                "ok",
            ],
        })

    def test_silver_columns_present(self):
        out, _meta = transform_silver(self._bronze())
        for col in SILVER_CONTRACT:
            assert col in out.columns, f"missing {col}"

    def test_no_risk_columns_in_silver(self):
        # Silver is the curated NLP layer; risk belongs to Gold.
        out, _meta = transform_silver(self._bronze())
        for col in GOLD_CONTRACT:
            assert col not in out.columns

    def test_redact_creates_comment_raw(self):
        out, _meta = transform_silver(
            self._bronze(),
            redact=True,
        )
        assert "Comment_raw" in out.columns
        assert "Comment" in out.columns
        # The original verbatim is preserved in Comment_raw.
        assert out["Comment_raw"].iloc[0] == "Great team, supportive manager"

    def test_redact_email(self):
        df = self._bronze()
        df.loc[0, "Comment"] = "Email me at john.doe@example.com about project ATLAS"
        out, _meta = transform_silver(df, redact=True)
        assert "[EMAIL]" in out["Comment"].iloc[0]
        assert "ATLAS" not in out["Comment"].iloc[0]

    def test_run_topics_false(self):
        out, _meta = transform_silver(self._bronze(), run_topics=False)
        # Without topics, Topic column should be NaN.
        assert out["Topic"].isna().all()

    def test_meta_has_topic_meta(self):
        _, meta = transform_silver(self._bronze())
        assert "topic_meta" in meta

    def test_no_risk_in_silver(self):
        out, _meta = transform_silver(self._bronze())
        # Belt-and-braces: the union of GOLD_CONTRACT should be
        # entirely absent from the Silver output.
        assert set(GOLD_CONTRACT).isdisjoint(set(out.columns))


# ---------- Gold -------------------------------------------------------

class TestGold:
    def _silver(self):
        return pd.DataFrame({
            "FeedbackID": ["F1", "F2", "F3"],
            "BU": ["Eng", "Eng", "Sales"],
            "Sentiment": ["positive", "negative", "No Comment"],
            "SentimentScore": [0.9, 0.95, 0.0],
            "Category1": ["Team", "Compensation and Benefits", "No Comment"],
            "AllCategories": ["Team(0.9)", "Compensation and Benefits(0.95)", ""],
            "Emotion": ["joy", "anger", "neutral"],
            "EmotionScore": [0.9, 0.85, 0.55],
            "Topic": [1, 2, -1],
            "TopicName": ["team", "compensation", "outlier"],
            "TopicKeywords": ["team,culture", "salary,bonus", ""],
            "EmployeeID": ["E1", "E1", "E2"],
            "SurveyDate": ["2025-01-15", "2025-04-15", "2025-04-15"],
            "Comment": ["x", "y", "z"],
        })

    def test_gold_columns_present(self):
        out, _meta = score_gold(self._silver())
        for col in GOLD_CONTRACT:
            assert col in out.columns, f"missing {col}"

    def test_risk_band_is_categorical(self):
        out, _meta = score_gold(self._silver())
        assert set(out["RiskBand"].unique()).issubset({"High", "Medium", "Low"})

    def test_risk_score_in_range(self):
        out, _meta = score_gold(self._silver())
        assert (out["RiskScore"] >= 0).all() and (out["RiskScore"] <= 100).all()

    def test_drops_comment_raw(self):
        silver = self._silver()
        silver["Comment_raw"] = silver["Comment"]
        out, _meta = score_gold(silver)
        # Comment_raw is dropped from the Gold output even if
        # present in Silver.
        assert "Comment_raw" not in out.columns

    def test_k_anonymity_suppresses(self):
        silver = self._silver()
        # Three rows: 2 in (Eng) and 1 in (Sales). With k=2, only
        # (Eng) survives; (Sales) is suppressed.
        out, _meta = score_gold(silver, k_anonymity=2)
        # The Sales row's CleanComment should be blanked. (We don't
        # have a CleanComment in this fixture, so check the
        # comment column itself.)
        sales = out[out["BU"] == "Sales"]
        assert (sales["Comment"] == "").all()

    def test_no_k_anonymity_by_default(self):
        silver = self._silver()
        out, _meta = score_gold(silver)
        # No suppression; all 3 comments preserved.
        assert (out["Comment"] != "").all()

    def test_type_error(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            score_gold([1, 2, 3])  # type: ignore[arg-type]


# ---------- Medallion (end-to-end) -------------------------------------

class TestMedallionEndToEnd:
    def _df(self):
        return pd.DataFrame({
            "FeedbackID": ["F1", "F2", "F3"],
            "BU": ["Eng", "Eng", "Sales"],
            "Comment": [
                "Great team, supportive manager",
                "Burned out, looking for new job, accepted an offer",
                "ok",
            ],
        })

    def test_returns_three_stages(self):
        stages = analyze_medallion(self._df())
        assert set(stages.keys()) >= {"bronze", "silver", "gold", "silver_meta", "gold_meta"}

    def test_stage_row_counts(self):
        stages = analyze_medallion(self._df())
        assert len(stages["bronze"]) == 3
        assert len(stages["silver"]) == 3
        assert len(stages["gold"]) == 3

    def test_silver_does_not_have_risk_columns(self):
        stages = analyze_medallion(self._df())
        for col in GOLD_CONTRACT:
            assert col not in stages["silver"].columns
        # And the bronze stage should have neither.
        for col in SILVER_CONTRACT:
            assert col not in stages["bronze"].columns

    def test_gold_has_silver_columns(self):
        stages = analyze_medallion(self._df())
        for col in SILVER_CONTRACT:
            assert col in stages["gold"].columns, f"Gold missing {col}"

    def test_redact_propagates(self):
        df = self._df()
        df.loc[0, "Comment"] = "Email me at john.doe@example.com"
        stages = analyze_medallion(df, redact=True)
        # Both Silver and Gold should have [EMAIL] in Comment.
        assert "[EMAIL]" in stages["silver"]["Comment"].iloc[0]
        assert "[EMAIL]" in stages["gold"]["Comment"].iloc[0]
        # Bronze has the original verbatim.
        assert "john.doe@example.com" in stages["bronze"]["Comment"].iloc[0]

    def test_rerun_gold_from_silver(self):
        """The whole point of the medaillon split: you can re-score
        Gold without re-running sentiment. We verify by inspecting
        that the Gold stage produces consistent RiskScores when run
        twice on the same Silver."""
        stages = analyze_medallion(self._df())
        gold_again, _ = score_gold(stages["silver"])
        # RiskScore must match the original gold (no sentiment
        # re-inference happens in score_gold).
        pd.testing.assert_series_equal(
            stages["gold"]["RiskScore"].reset_index(drop=True),
            gold_again["RiskScore"].reset_index(drop=True),
            check_names=False,
        )


# ---------- Public-API surface -----------------------------------------

class TestPublicApiExports:
    def test_medallion_functions_exported(self):
        for name in ("analyze_medallion", "ingest_bronze", "transform_silver",
                     "score_gold", "BRONZE_CONTRACT", "SILVER_CONTRACT",
                     "GOLD_CONTRACT"):
            assert hasattr(eva, name), f"missing {name!r}"

    def test_contracts_contain_expected_columns(self):
        assert "Comment" in BRONZE_CONTRACT
        assert "Sentiment" in SILVER_CONTRACT
        assert "RiskScore" in GOLD_CONTRACT
        # Bronze and Silver and Gold should be disjoint unions.
        assert set(SILVER_CONTRACT).isdisjoint(set(BRONZE_CONTRACT))
        assert set(GOLD_CONTRACT).isdisjoint(set(SILVER_CONTRACT))
