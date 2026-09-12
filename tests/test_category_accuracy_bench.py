"""Tests for the category accuracy bench.

These are fast (~30s on a CPU) — they exercise the bench harness
end-to-end with the synthetic data so that:
  1. The bench produces a CSV with three rows (BART zero-shot,
     distilbert zero-shot, SetFit few-shot).
  2. The accuracy numbers are in a sane range (0 < acc < 1,
     n_matched <= n_scored).
  3. The SetFit model actually loaded and ran. This catches the
     failure mode where SetFit's Trainer API changes between
     versions and the bench silently breaks.

Run with:
    pytest -m accuracy_bench
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.accuracy_bench

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "benchmarks" / "results_category_accuracy.csv"


class TestCategoryAccuracyBench:
    def test_bench_runs_and_writes_csv(self, tmp_path):
        """Run the accuracy bench against a tmp CSV. We don't
        touch the committed CSV here — that's the slow path."""
        try:
            import transformers  # noqa: F401
            import setfit  # noqa: F401
        except ImportError:
            pytest.skip("transformers or setfit not installed")

        bench_csv = tmp_path / "accuracy.csv"
        bench_md = tmp_path / "accuracy.md"

        result = subprocess.run(
            [
                sys.executable, "benchmarks/bench_category_accuracy.py",
                "--rows", "20",  # tiny for CI
                "--csv", str(bench_csv),
                "--out", str(bench_md),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "EMPLOYEE_VOICE_ALLOW_FALLBACK": "1", "PYTHONPATH": "."},
        )
        assert result.returncode == 0, (
            f"bench failed: stderr:\n{result.stderr[-1500:]}"
        )
        assert bench_csv.exists(), "CSV not written"
        df = pd.read_csv(bench_csv)
        # Three approaches: bart zero-shot, distilbert zero-shot, SetFit few-shot.
        assert len(df) == 3, f"expected 3 rows, got {len(df)}"
        for _, row in df.iterrows():
            assert 0 < row["accuracy"] <= 1.0, (
                f"row {row['model']!r} has accuracy={row['accuracy']}"
            )
            assert 0 < row["n_scored"], (
                f"row {row['model']!r} has n_scored={row['n_scored']} — "
                f"the bench may have skipped too many rows"
            )
            assert row["n_matched"] <= row["n_scored"]
            assert row["elapsed_sec"] > 0
