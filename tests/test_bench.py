"""In-process smoke benchmark.

The full benchmark harness in `benchmarks/bench.py` measures
throughput at 1k-50k rows. This test is the fast version that
runs inside the regular pytest suite — it just checks we don't
have a serious regression in the layer runner.

Run with:
    pytest tests/test_bench.py -v
    pytest -m "not bench"   # skip on slow CI
"""
from __future__ import annotations
import os
import time

import pytest

# Force fallback engines.
os.environ.setdefault("EMPLOYEE_VOICE_ALLOW_FALLBACK", "1")
import employee_voice.config as _cfg  # noqa: E402
_cfg.ALLOW_FALLBACK = True
from employee_voice import analyzers as _analyzers  # noqa: E402
_analyzers._reset_backends()

import employee_voice  # noqa: E402


pytestmark = pytest.mark.bench


def _run_pipeline(n: int) -> float:
    df = employee_voice.generate_synthetic(n=n, seed=1)
    t0 = time.perf_counter()
    employee_voice.analyze_feedback(df, run_topics=True)
    return time.perf_counter() - t0


def test_lexicon_pipeline_200_rows_under_threshold():
    """A 200-row lexicon pipeline should finish well under 10 seconds
    on any reasonable machine. The threshold is loose on purpose:
    it's a regression guard, not a precise SLA."""
    elapsed = _run_pipeline(200)
    assert elapsed < 10.0, f"pipeline too slow: {elapsed:.2f}s for 200 rows"


def test_score_gold_from_silver_faster_than_full_pipeline():
    """The medaillon payoff: re-scoring Gold from Silver should be
    faster than running the full pipeline from Bronze. This is the
    core design promise."""
    df = employee_voice.generate_synthetic(n=500, seed=2)
    silver, _ = employee_voice.medallion.transform_silver(df)

    t0 = time.perf_counter()
    employee_voice.analyze_feedback(df, run_topics=True)
    full_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    employee_voice.score_gold(silver)
    gold_time = time.perf_counter() - t0

    # We expect at least a 2x speedup at 500 rows on the lexicon
    # backend. (The full bench measures 11x at 10k rows.)
    assert gold_time < full_time / 2, (
        f"Gold re-score ({gold_time:.3f}s) should be at least 2x faster "
        f"than the full pipeline ({full_time:.3f}s)"
    )
