"""Tests for the synthetic data generator."""
from __future__ import annotations
import pandas as pd
import pytest

from employee_voice.synth_data import generate_synthetic_dataset, PERSONAS


class TestSyntheticData:
    def test_default_size(self):
        df = generate_synthetic_dataset(n=100)
        assert len(df) == 100

    def test_required_columns(self):
        df = generate_synthetic_dataset(n=50)
        for c in ("FeedbackID", "EmployeeID", "SurveyDate", "SurveyType",
                  "BU", "Dept", "EmployeeGroup", "Comment"):
            assert c in df.columns

    def test_deterministic(self):
        a = generate_synthetic_dataset(n=50, seed=123)
        b = generate_synthetic_dataset(n=50, seed=123)
        assert (a["Comment"].tolist() == b["Comment"].tolist())

    def test_seeds_differ(self):
        a = generate_synthetic_dataset(n=50, seed=1)
        b = generate_synthetic_dataset(n=50, seed=2)
        # With different seeds the Comment series should differ somewhere.
        assert not (a["Comment"].tolist() == b["Comment"].tolist())

    def test_dates_span_four_quarters(self):
        df = generate_synthetic_dataset(n=200, seed=42)
        dates = pd.to_datetime(df["SurveyDate"])
        assert dates.dt.year.min() == 2025
        # All 4 quarters should appear.
        assert set(dates.dt.quarter.unique()) == {1, 2, 3, 4}

    def test_edge_cases_present(self):
        # Run many rows; expect at least one "no comment" / empty / PII edge.
        df = generate_synthetic_dataset(n=500, seed=42)
        comments = df["Comment"].astype(str).tolist()
        assert any("no comment" in c.lower() for c in comments)
        assert any("example.com" in c for c in comments)
        assert any("ATLAS" in c for c in comments)
        assert any("Sarah Johnson" in c for c in comments)

    def test_all_personas_represented(self):
        df = generate_synthetic_dataset(n=500, seed=42)
        # No assertion on the exact split, but every persona's signature
        # comment should appear at least once.
        for persona in PERSONAS.values():
            assert any(
                any(c in str(row) for c in persona["comments"])
                for row in df["Comment"]
            )
