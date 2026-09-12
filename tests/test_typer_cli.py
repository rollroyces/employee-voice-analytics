"""Tests for the Typer CLI surface (no Rich rendering, no subprocess)."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from employee_voice.typer_cli import app, _render_dashboard

runner = CliRunner()


class TestVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "employee-voice-analytics" in result.stdout


class TestGenerateSample:
    def test_generates_csv(self, tmp_path):
        out = tmp_path / "sample.csv"
        result = runner.invoke(app, ["generate-sample", "--rows", "20", "--out", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 20
        for c in ("FeedbackID", "BU", "Comment"):
            assert c in df.columns


class TestAnalyze:
    def test_runs_on_sample(self, tmp_path):
        outdir = tmp_path / "out"
        result = runner.invoke(app, [
            "analyze",
            "--input", "data/sample_feedback.csv",
            "--outdir", str(outdir),
        ])
        assert result.exit_code == 0
        for f in ("fact_employee_feedback.csv", "dim_topic.csv",
                  "summary_category.csv", "summary_bu.csv"):
            assert (outdir / f).exists()

    def test_missing_input_returns_error(self):
        result = runner.invoke(app, [
            "analyze", "--input", "nonexistent.csv", "--outdir", "/tmp/x"
        ])
        assert result.exit_code != 0

    def test_no_topics_flag(self, tmp_path):
        outdir = tmp_path / "out"
        result = runner.invoke(app, [
            "analyze",
            "--input", "data/sample_feedback.csv",
            "--outdir", str(outdir),
            "--no-topics",
        ])
        assert result.exit_code == 0
        fact = pd.read_csv(outdir / "fact_employee_feedback.csv")
        assert fact["Topic"].isna().all()


class TestRedact:
    def test_redacts_csv(self, tmp_path):
        src = tmp_path / "in.csv"
        out = tmp_path / "out.csv"
        src.write_text(
            "Comment\n"
            "Email me at john.doe@example.com about project ATLAS\n"
            "My manager Sarah Johnson is great\n"
        )
        result = runner.invoke(app, [
            "redact", "--input", str(src), "--out", str(out),
        ])
        assert result.exit_code == 0
        df = pd.read_csv(out)
        assert "[EMAIL]" in df["Comment_redacted"].iloc[0]
        assert "[MANAGER_NAME]" in df["Comment_redacted"].iloc[1]


class TestDashboard:
    def test_dashboard_renders(self):
        from io import StringIO
        from rich.console import Console
        df = pd.DataFrame({
            "FeedbackID": ["F1", "F2", "F3"],
            "Sentiment": ["positive", "negative", "neutral"],
            "RiskBand": ["Low", "High", "Medium"],
            "RiskScore": [10.0, 95.0, 50.0],
            "TopicName": ["team / culture", "manager / compensation", "workload"],
            "BU": ["Eng", "Eng", "Sales"],
        })
        # Redirect Rich's Console to a buffer so we can assert on the output.
        from employee_voice import typer_cli
        buf = StringIO()
        typer_cli.console = Console(file=buf, force_terminal=False)
        _render_dashboard(df, by="BU", top_n=5)
        out = buf.getvalue()
        assert "Sentiment distribution" in out
        assert "Risk band" in out
        assert "Eng" in out
