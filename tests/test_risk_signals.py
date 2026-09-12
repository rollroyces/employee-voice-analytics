"""Tests for push/pull factors and risk velocity."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from employee_voice.risk_signals import (
    detect_factors,
    add_factor_columns,
    compute_risk_velocity,
    PUSH_FACTORS,
    PULL_FACTORS,
)


class TestDetectFactors:
    def test_burnout_push(self):
        assert "burnout" in detect_factors("I'm burned out and exhausted", PUSH_FACTORS)

    def test_intent_to_leave_pull(self):
        assert "intent_to_leave" in detect_factors("I quit, last day next month", PULL_FACTORS)

    def test_external_offer_pull(self):
        assert "external_offer" in detect_factors("I accepted an offer elsewhere", PULL_FACTORS)

    def test_no_match(self):
        assert detect_factors("Just a status update", PUSH_FACTORS) == []

    def test_multiple_push_factors(self):
        text = "Burned out, underpaid, my manager is unresponsive"
        hits = detect_factors(text, PUSH_FACTORS)
        assert set(hits) == {"burnout", "compensation", "manager"}

    def test_empty_text(self):
        assert detect_factors("", PUSH_FACTORS) == []


class TestAddFactorColumns:
    def test_columns_added(self):
        df = pd.DataFrame({
            "Comment": ["Burned out and exhausted", "I accepted an offer"],
        })
        out = add_factor_columns(df)
        assert "PushFactors" in out.columns
        assert "PullFactors" in out.columns

    def test_factors_detected(self):
        df = pd.DataFrame({
            "Comment": ["Burned out, looking for new job. I accepted an offer."],
        })
        out = add_factor_columns(df)
        assert "burnout" in out["PushFactors"].iloc[0]
        assert "job_search" in out["PullFactors"].iloc[0]
        assert "external_offer" in out["PullFactors"].iloc[0]

    def test_uses_clean_comment_when_present(self):
        df = pd.DataFrame({
            "Comment": ["ORIGINAL"],
            "CleanComment": ["Burned out and exhausted"],
        })
        out = add_factor_columns(df)
        assert "burnout" in out["PushFactors"].iloc[0]

    def test_missing_comment_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        out = add_factor_columns(df)
        assert (out["PushFactors"] == "").all()
        assert (out["PullFactors"] == "").all()


class TestRiskVelocity:
    def test_no_employee_id_returns_null(self):
        df = pd.DataFrame({
            "SurveyDate": ["2026-01-15", "2026-04-15"],
            "Sentiment": ["negative", "negative"],
        })
        out = compute_risk_velocity(df)
        assert out["RiskVelocity"].isna().all()

    def test_no_survey_date_returns_null(self):
        df = pd.DataFrame({
            "EmployeeID": ["E1", "E1"],
            "Sentiment": ["negative", "negative"],
        })
        out = compute_risk_velocity(df)
        assert out["RiskVelocity"].isna().all()

    def test_employee_getting_more_negative(self):
        df = pd.DataFrame({
            "EmployeeID": ["E1", "E1", "E1", "E1"],
            "SurveyDate": [
                "2025-10-01", "2025-10-02",   # Q4 2025 — both negative
                "2026-01-15", "2026-02-20",   # Q1 2026 — both negative
            ],
            "Sentiment": ["negative", "negative", "negative", "negative"],
        })
        out = compute_risk_velocity(df)
        # First 2 rows are Q4 2025 -> no prior, NaN.
        # Q1 2026 rate = 1.0, Q4 2025 rate = 1.0, velocity = 0
        assert out["RiskVelocity"].iloc[:2].isna().all()
        assert (out["RiskVelocity"].iloc[2:] == 0.0).all()

    def test_velocity_clipped_to_one(self):
        df = pd.DataFrame({
            "EmployeeID": ["E1"] * 4,
            "SurveyDate": [
                "2025-10-01", "2025-10-02",
                "2026-01-15", "2026-02-20",
            ],
            "Sentiment": ["positive", "positive", "negative", "negative"],
        })
        out = compute_risk_velocity(df)
        # First 2 rows are Q4 2025 -> no prior quarter, so NaN.
        # Q1 2026 rate = 1.0, Q4 2025 rate = 0.0, velocity = +1.0 (clipped)
        assert out["RiskVelocity"].iloc[:2].isna().all()
        assert (out["RiskVelocity"].iloc[2:] == 1.0).all()

    def test_velocity_per_employee(self):
        df = pd.DataFrame({
            "EmployeeID": ["E1", "E1", "E2", "E2"],
            "SurveyDate": ["2025-10-01", "2026-01-15", "2025-10-01", "2026-01-15"],
            "Sentiment": ["positive", "negative", "negative", "positive"],
        })
        out = compute_risk_velocity(df)
        # E1: went positive -> negative => +1.0
        e1_velocity = out[out["EmployeeID"] == "E1"]["RiskVelocity"].iloc[-1]
        # E2: went negative -> positive => -1.0
        e2_velocity = out[out["EmployeeID"] == "E2"]["RiskVelocity"].iloc[-1]
        assert e1_velocity == 1.0
        assert e2_velocity == -1.0
