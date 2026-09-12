"""Tests for the privacy scrubber and k-anonymity filter."""
from __future__ import annotations
import os
import sys

import pandas as pd
import pytest

# Skip Presidio-dependent tests if the model isn't installed.
presidio_available = True
try:
    from presidio_analyzer import AnalyzerEngine  # noqa: F401
except ImportError:
    presidio_available = False
try:
    import spacy
    spacy.util.get_installed_models()  # returns list of available model names
    if not any(m.startswith("en_core_web_") for m in spacy.util.get_installed_models()):
        presidio_available = False
except (ImportError, OSError):
    presidio_available = False

from employee_voice.privacy import (
    PIIScrubber,
    KAnonymityConfig,
    apply_k_anonymity,
)


# ---------- regex backend ---------------------------------------------

class TestRegexPIIScrubber:
    def setup_method(self):
        self.scrubber = PIIScrubber(backend="regex")

    def test_email_is_redacted(self):
        r = self.scrubber.scrub("Email me at john.doe@example.com please")
        assert r.text == "Email me at [EMAIL] please"
        assert r.found["EMAIL"] == 1

    def test_phone_is_redacted(self):
        r = self.scrubber.scrub("Call +1 555-123-4567 or 555.123.4567")
        assert "[PHONE]" in r.text
        assert r.found.get("PHONE", 0) >= 1

    def test_ssn_is_redacted(self):
        r = self.scrubber.scrub("My SSN is 123-45-6789 by the way")
        assert "[SSN]" in r.text

    def test_employee_id_is_redacted(self):
        r = self.scrubber.scrub("employee id EMP12345 reported the bug")
        assert "[EMPLOYEE_ID]" in r.text

    def test_manager_name_via_my_manager(self):
        r = self.scrubber.scrub("My manager Sarah Johnson is great")
        assert "[MANAGER_NAME]" in r.text

    def test_manager_name_via_reports_to(self):
        r = self.scrubber.scrub("I report to John Smith on Tuesdays")
        assert "[MANAGER_NAME]" in r.text

    def test_colleague_name_is_employee(self):
        r = self.scrubber.scrub("Loved working with Alice on the launch")
        assert "[EMPLOYEE_NAME]" in r.text

    def test_project_codename_redacted(self):
        r = self.scrubber.scrub("Great work on project ATLAS this quarter")
        assert "[PROJECT_CODE]" in r.text
        assert "ATLAS" not in r.text

    def test_no_pii_returns_unchanged(self):
        text = "Compensation is below market and burnout is real"
        r = self.scrubber.scrub(text)
        assert r.text == text
        assert r.found == {}
        assert r.redacted_count == 0

    def test_empty_string(self):
        r = self.scrubber.scrub("")
        assert r.text == ""
        assert r.redacted_count == 0

    def test_none_safe(self):
        r = self.scrubber.scrub(None)  # type: ignore[arg-type]
        assert r.text == ""
        assert r.redacted_count == 0

    def test_multiple_entities_in_one_string(self):
        r = self.scrubber.scrub(
            "Email john@example.com, call 555-123-4567, "
            "and ask manager Sarah Johnson about project ATLAS"
        )
        assert r.found.get("EMAIL", 0) == 1
        assert r.found.get("PHONE", 0) == 1
        assert r.found.get("MANAGER_NAME", 0) == 1
        assert r.found.get("PROJECT_CODE", 0) == 1

    def test_scrub_series(self):
        s = pd.Series([
            "Email john@example.com",
            "no PII here",
            "Manager Bob is great",
        ])
        out = self.scrubber.scrub_series(s)
        assert "[EMAIL]" in out.iloc[0]
        assert out.iloc[1] == "no PII here"
        assert "[MANAGER_NAME]" in out.iloc[2]

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown PII backend"):
            PIIScrubber(backend="magic")


# ---------- Presidio backend ------------------------------------------

@pytest.mark.skipif(not presidio_available, reason="presidio + spaCy NER model not installed")
class TestPresidioPIIScrubber:
    def test_email_and_phone_detected(self):
        s = PIIScrubber(backend="presidio")
        r = s.scrub("Email me at john.doe@example.com or call +1 555-123-4567")
        assert r.found.get("EMAIL", 0) >= 1
        assert r.found.get("PHONE", 0) >= 1

    def test_manager_context_promotion(self):
        s = PIIScrubber(backend="presidio")
        r = s.scrub("My manager Sarah Johnson is great")
        assert r.found.get("MANAGER_NAME", 0) >= 1
        assert "[MANAGER_NAME]" in r.text

    def test_employee_id_custom_recogniser(self):
        s = PIIScrubber(backend="presidio")
        r = s.scrub("See ticket from EMP998877 for context")
        assert r.found.get("EMPLOYEE_ID", 0) >= 1


# ---------- k-anonymity -----------------------------------------------

class TestKAnonymity:
    def test_under_threshold_suppresses_verbatim(self):
        df = pd.DataFrame({
            "BU": ["Eng"] * 3 + ["Sales"] * 10,
            "Dept": ["A"] * 3 + ["X"] * 5 + ["Y"] * 5,
            "Comment": [f"c{i}" for i in range(13)],
            "RiskScore": list(range(13)),
        })
        out, summary = apply_k_anonymity(df, ["BU", "Dept"], KAnonymityConfig(threshold=5))
        # Eng/A slice (3 rows) should have blank Comments.
        assert (out.loc[out["BU"] == "Eng", "Comment"] == "").all()
        # Sales slices (5 each) should be untouched.
        assert (out.loc[(out["BU"] == "Sales") & (out["Dept"] == "X"), "Comment"] != "").all()
        # Risk scores preserved either way (for aggregate dashboards).
        assert out["RiskScore"].sum() == df["RiskScore"].sum()
        # Slice summary has one row per slice.
        assert len(summary) == 3
        assert (summary["slice_count"] < 5).sum() == 1

    def test_all_slices_above_threshold_no_suppression(self):
        df = pd.DataFrame({
            "BU": ["A", "A", "A", "B", "B", "B"],
            "Comment": ["x"] * 6,
        })
        out, summary = apply_k_anonymity(df, ["BU"], KAnonymityConfig(threshold=3))
        assert (out["Comment"] != "").all()
        assert len(summary) == 2

    def test_missing_group_column_raises(self):
        df = pd.DataFrame({"Comment": ["x"]})
        with pytest.raises(ValueError, match="not in DataFrame"):
            apply_k_anonymity(df, ["NopeNopeNope"], KAnonymityConfig(threshold=5))
