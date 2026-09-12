"""Smoke test for the category-model benchmark.

Runs the bench with a tiny row count so the test is fast (~30s)
and doesn't depend on the full HF model cache. The point is to
catch three classes of regression in CI:

  1. The parser logic broke (e.g. someone changed parts[2] back
     to parts[2] in the parent parser). This is the bug we
     already shipped once; this test pins the parser format
     with a live subprocess.
  2. The harness itself regresses (e.g. subprocess.run times out,
     the child fails to import employee_voice, the CSV doesn't
     get written).
  3. All rows in the CSV end up with elapsed_sec=0 (e.g. the
     warmup eats the measurement) — a silent failure mode
     that wouldn't show up until someone actually looked at the
     numbers.

Run with:
    pytest -m bench_smoke
    # or explicitly:
    pytest tests/test_bench_smoke.py -v
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.bench_smoke

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "benchmarks" / "results_category_models.csv"
MD_PATH = REPO_ROOT / "benchmarks" / "RESULTS_category_models.md"


def _run_bench(rows: int = 5, timeout_s: int = 600) -> None:
    """Run the bench in-process. Skips if transformers isn't installed."""
    # The category bench needs the [ml] extra.
    try:
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed; install [ml] extra")

    # --models is a single comma-separated flag (NOT a repeatable
    # one) in the bench. Multiple --models on the CLI would
    # silently overwrite each other and only the last one would run.
    models = ",".join([
        "facebook/bart-large-mnli",
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "cross-encoder/nli-deberta-v3-small",
        "typeform/distilbert-base-uncased-mnli",
    ])

    result = subprocess.run(
        [
            sys.executable, "benchmarks/bench_category_models.py",
            "--rows", str(rows),
            "--models", models,
            "--csv", str(CSV_PATH),
            "--out", str(MD_PATH),
            "--timeout-s", str(timeout_s),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s + 60,  # buffer for parent startup
        env={**os.environ, "EMPLOYEE_VOICE_ALLOW_FALLBACK": "1", "PYTHONPATH": "."},
    )
    assert result.returncode == 0, (
        f"bench exited {result.returncode}.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )


class TestCategoryBenchSmoke:
    def test_bench_writes_csv(self, tmp_path):
        """Run the bench in a tmp dir so we don't overwrite the
        committed CSV. The point of the smoke test is to catch
        regressions; we don't want to commit noise."""
        bench_csv = tmp_path / "category_models.csv"
        bench_md = tmp_path / "category_models.md"

        # --models is a single comma-separated flag, not a
        # repeatable one. Use 2 models for the tmp test (faster).
        models = "facebook/bart-large-mnli,typeform/distilbert-base-uncased-mnli"

        result = subprocess.run(
            [
                sys.executable, "benchmarks/bench_category_models.py",
                "--rows", "5",
                "--models", models,
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
            f"bench failed: {result.stderr[-1500:]}"
        )
        assert bench_csv.exists(), f"CSV not written. stdout:\n{result.stdout}"
        df = pd.read_csv(bench_csv)
        assert len(df) == 2, f"expected 2 rows in CSV, got {len(df)}"
        # Both rows have positive elapsed time — the silent-failure
        # mode where elapsed_sec=0 would slip through without this.
        for _, row in df.iterrows():
            assert row["elapsed_sec"] > 0, (
                f"row {row['label']!r} has elapsed_sec={row['elapsed_sec']} — "
                f"the warmup is probably eating the measurement"
            )
            assert row["rows_per_sec"] > 0, (
                f"row {row['label']!r} has rows_per_sec={row['rows_per_sec']}"
            )
        # rows_per_sec should be roughly rows / elapsed. Allow a 2x
        # slack to account for subprocess / interpreter overhead
        # the parent excludes via the MEASURE line.
        for _, row in df.iterrows():
            expected = 5 / row["elapsed_sec"]
            assert 0.5 * expected <= row["rows_per_sec"] <= 2 * expected, (
                f"row {row['label']!r}: rows_per_sec={row['rows_per_sec']:.2f} "
                f"doesn't match rows/elapsed={expected:.2f}"
            )

    def test_bench_against_committed_csv(self):
        """Run the bench against the committed CSV/MD paths so CI
        catches regressions to the actual artifacts. The committed
        CSV gets overwritten — that's intentional; the smoke test
        is supposed to *produce* a fresh measurement.
        """
        try:
            _run_bench(rows=5, timeout_s=600)
        except subprocess.TimeoutExpired:
            pytest.fail("bench exceeded 10 minutes; the harness is broken")
        # CSV must exist and have 4 rows (one per model).
        assert CSV_PATH.exists(), f"committed CSV not at {CSV_PATH}"
        df = pd.read_csv(CSV_PATH)
        assert len(df) == 4, (
            f"expected 4 rows in {CSV_PATH}, got {len(df)}: {df['label'].tolist()}"
        )
        # All rows have non-NaN, positive elapsed time.
        for _, row in df.iterrows():
            elapsed = float(row["elapsed_sec"])
            assert elapsed > 0, (
                f"row {row['label']!r} has elapsed_sec={elapsed} — "
                f"this is the silent-zero-elapsed failure mode"
            )

    def test_markdown_internally_consistent(self):
        """The MD report is generated by the bench itself. If it's
        generated at all, the speedup claim should be derivable
        from the CSV. This catches the case where the bench
        'succeeds' but writes a nonsense markdown."""
        try:
            _run_bench(rows=5, timeout_s=600)
        except subprocess.TimeoutExpired:
            pytest.fail("bench exceeded 10 minutes")
        md = MD_PATH.read_text()
        # The MD must mention all four models and the words "rows/sec".
        for model_label in [
            "bart-large-mnli", "deberta-v3-base", "deberta-v3-small",
            "distilbert-base-mnli",
        ]:
            assert model_label in md, (
                f"MD report missing {model_label!r}\nfull report:\n{md}"
            )
        assert "rows/sec" in md
