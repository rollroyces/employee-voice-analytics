"""
Performance benchmark for the Employee Voice Analytics design.

Measures each layer in isolation and the end-to-end medaillon flow
at 1k, 10k, and 50k rows. Runs on a single CPU (no Spark), so the
absolute numbers are machine-specific, but the *ratios* between
backends and between pipeline stages are what matters for design
decisions.

Run with:
    python benchmarks/bench.py [--rows 1000,10000,50000] [--backend lexicon|hf] [--out FILE]

Outputs a CSV with one row per (rows, layer) combination and prints
a summary table. Writes a markdown report to `benchmarks/RESULTS.md`
if --out is given.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

# Force a specific backend before any employee_voice imports.
os.environ.setdefault("EMPLOYEE_VOICE_ALLOW_FALLBACK", "1")
import pandas as pd

import employee_voice  # noqa: E402
import employee_voice.config as _cfg  # noqa: E402
from employee_voice import preprocess, analyzers, medallion  # noqa: E402
from employee_voice.analyzers import (  # noqa: E402
    analyze_sentiment, analyze_category, analyze_emotion,
)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    label: str
    rows: int
    backend: str
    seconds: float
    peak_mem_mb: float
    extras: dict = field(default_factory=dict)

    @property
    def rows_per_sec(self) -> float:
        return self.rows / self.seconds if self.seconds > 0 else float("inf")


def _time_and_mem(fn: Callable[[], None], label: str, rows: int, backend: str) -> BenchResult:
    """Run `fn` once, measuring wall time + peak heap via tracemalloc.

    We run once (not averaged) because the HF / TF-IDF backends
    are too slow at 50k rows to repeat; for fast layers we still
    trust the single measurement since variance is dominated by
    OS noise at these scales.
    """
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        fn()
    finally:
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return BenchResult(
        label=label, rows=rows, backend=backend,
        seconds=elapsed, peak_mem_mb=peak / 1024 / 1024,
    )


# ---------------------------------------------------------------------------
# Per-layer benchmarks
# ---------------------------------------------------------------------------
def bench_preprocess(df, backend):
    def run():
        preprocess.clean_text(df["Comment"].iloc[0])
        df["Comment"].fillna("").astype(str).map(preprocess.clean_text)
    return _time_and_mem(run, "preprocess", len(df), backend)


def bench_pii_regex(df, backend):
    def run():
        scrubber = employee_voice.PIIScrubber(backend="regex")
        scrubber.scrub_series(df["Comment"].fillna("").astype(str))
    return _time_and_mem(run, "pii_scrub_regex", len(df), backend)


def bench_sentiment(df, backend):
    def run():
        # Force the chosen backend. HF: on a Mac without CUDA the
        # CPU pipeline is the bottleneck; we still measure it for
        # the design's sake.
        analyze_sentiment(df["Comment"].fillna("").tolist())
    return _time_and_mem(run, "sentiment", len(df), backend)


def bench_category(df, backend):
    def run():
        analyze_category(df["Comment"].fillna("").tolist())
    return _time_and_mem(run, "category", len(df), backend)


def bench_emotion(df, backend):
    def run():
        analyze_emotion(df["Comment"].fillna("").tolist())
    return _time_and_mem(run, "emotion", len(df), backend)


def bench_risk(df, backend):
    """Layer 5 only — uses a pre-built Silver table so we measure
    just the risk + push/pull + velocity + k-anonymity path, not
    the sentiment/category/emotion layers.
    """
    # Build a minimal Silver table with everything set to fixed
    # values so the risk layer has the columns it expects. We
    # don't go through transform_silver because we want the
    # risk-only time, not Silver's full stack.
    silver = df.copy()
    silver["Sentiment"] = "negative"
    silver["SentimentScore"] = 0.8
    silver["Category1"] = "Compensation and Benefits"
    silver["AllCategories"] = "Compensation and Benefits(0.8)"
    silver["Emotion"] = "anger"
    silver["EmotionScore"] = 0.7
    silver["CleanComment"] = silver["Comment"].fillna("").astype(str)

    def run():
        medallion.score_gold(silver)
    return _time_and_mem(run, "score_gold_layer5", len(df), backend)


def bench_topics(df, backend):
    def run():
        employee_voice.analyze_feedback(df, run_topics=True)
    return _time_and_mem(run, "themes", len(df), backend)


def bench_full_pipeline(df, backend):
    def run():
        employee_voice.analyze_feedback(df, run_topics=True)
    return _time_and_mem(run, "analyze_feedback", len(df), backend)


def bench_medallion_gold_from_silver(silver, backend):
    """The medaillon payoff: re-score Gold without re-running sentiment.
    We measure the Gold stage alone, having prepared Silver in a
    separate (untimed) setup step.
    """
    def run():
        medallion.score_gold(silver)
    return _time_and_mem(run, "score_gold_from_silver", len(silver), backend)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def make_dataset(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset of n rows. The persona mix varies
    so we get realistic distributions of positive / negative /
    neutral comments (a purely-positive corpus would be unrealistically
    easy to classify).
    """
    return employee_voice.generate_synthetic(n=n, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1000,10000",
                    help="Comma-separated row counts to benchmark.")
    ap.add_argument("--backend", choices=["lexicon", "hf", "auto"],
                    default="lexicon",
                    help="Analyzer backend. 'lexicon' is fast; 'hf' is "
                         "accurate but slow on CPU. 'auto' uses whatever "
                         "is installed.")
    ap.add_argument("--out", default=None,
                    help="Write a markdown report to this path.")
    ap.add_argument("--csv", default=None,
                    help="Write a CSV of raw measurements to this path.")
    args = ap.parse_args()

    # Resolve backend. The HF mode is "force HuggingFace for the
    # three HF-using layers (sentiment / category / emotion)" but
    # we still use the lexicon / TF-IDF fallback for the themes
    # layer — BERTopic would take 5+ minutes to download and isn't
    # part of what we're benchmarking.
    if args.backend == "lexicon":
        _cfg.ALLOW_FALLBACK = True
        analyzers._reset_backends()
        backend = "lexicon"
    elif args.backend == "hf":
        _cfg.ALLOW_FALLBACK = True  # themes layer is exempt
        analyzers._reset_backends()
        # Force HF for sentiment / category / emotion by toggling
        # the per-layer _BACKENDS cache.
        from employee_voice.analyzers import _Backends
        import employee_voice.analyzers as _a
        _a._BACKENDS = _Backends(
            sentiment="transformers", category="transformers", emotion="transformers",
        )
        backend = "hf+lexicon-themes"
    else:
        backend = "hf+lexicon-themes" if not _cfg.ALLOW_FALLBACK else "lexicon"

    rows_list = [int(s) for s in args.rows.split(",")]
    results: List[BenchResult] = []

    for n in rows_list:
        print(f"\n=== Benchmarking n = {n:,} rows (backend={backend}) ===")
        df = make_dataset(n, seed=42)
        print(f"  dataset built ({len(df)} rows)")

        # Per-layer (cheap layers first so we don't have to wait 5
        # minutes for an HF 50k sentiment run).
        results.append(bench_preprocess(df, backend))
        results.append(bench_pii_regex(df, backend))
        results.append(bench_full_pipeline(df, backend))
        if n <= 10000:
            results.append(bench_sentiment(df, backend))
            results.append(bench_category(df, backend))
            results.append(bench_emotion(df, backend))
            results.append(bench_risk(df, backend))
            results.append(bench_topics(df, backend))
        else:
            print(f"  (skipping per-layer HF benches at n={n} — full-pipeline bench is representative)")

        # Medaillon payoff: re-score Gold from the same Silver.
        # We need a real Silver; build it once with the full pipeline.
        # The first full-pipeline call above produced the fact table
        # but didn't expose it; re-run once and capture Silver.
        silver_df, _ = medallion.transform_silver(df)
        gold_a, _ = medallion.score_gold(silver_df)
        gold_b, _ = medallion.score_gold(silver_df)
        # Medaillon payoff invariant: re-scoring Gold from the same
        # Silver must be deterministic. We assert this in the bench
        # so we'd notice immediately if a future change to the risk
        # layer breaks the property.
        a_scores = list(gold_a["RiskScore"].astype(float))
        b_scores = list(gold_b["RiskScore"].astype(float))
        if a_scores != b_scores:
            raise AssertionError(
                "Gold re-score is not deterministic — investigate before "
                "publishing the medaillon payoff claim."
            )
        results.append(bench_medallion_gold_from_silver(silver_df, backend))

    # Print summary.
    print("\n" + "=" * 80)
    print(f"Results — backend = {backend}, machine = {platform.machine()}, "
          f"python = {platform.python_version()}")
    print("=" * 80)
    print(f"{'layer':30s} {'rows':>8s} {'sec':>10s} {'rows/s':>10s} {'peak MB':>10s}")
    print("-" * 80)
    for r in results:
        print(f"{r.label:30s} {r.rows:>8,d} {r.seconds:>10.3f} "
              f"{r.rows_per_sec:>10,.0f} {r.peak_mem_mb:>10.1f}")
    print("=" * 80)

    # CSV
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        rows_csv = [vars(r) for r in results]
        pd.DataFrame(rows_csv).to_csv(args.csv, index=False)
        print(f"\nWrote {len(results)} raw measurements to {args.csv}")

    # Markdown report
    if args.out:
        write_markdown_report(results, args.out, backend=backend,
                              rows_list=rows_list)
        print(f"Wrote markdown report to {args.out}")


def write_markdown_report(results: List[BenchResult], path: str,
                          backend: str, rows_list: List[int]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Employee Voice Analytics — benchmark report",
        "",
        f"_Generated by `benchmarks/bench.py` on {platform.machine()}, "
        f"Python {platform.python_version()}, backend = `{backend}`._",
        "",
        "Raw numbers — single run, no averaging. Variance is dominated by "
        "OS / GC noise, not measurement error, so single samples are "
        "informative enough for design decisions.",
        "",
        "| layer | rows | seconds | rows/sec | peak MB |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r.label}` | {r.rows:,} | {r.seconds:.3f} | "
            f"{r.rows_per_sec:,.0f} | {r.peak_mem_mb:.1f} |"
        )
    lines += [
        "",
        "## Medaillon payoff",
        "",
        "Re-scoring Gold from the same Silver (changing k-anonymity, "
        "risk weights, push/pull keywords) should be much cheaper than "
        "Bronze→Silver because Layers 1-4 (sentiment, category, emotion, "
        "themes) are not re-run. The `score_gold_from_silver` row above "
        "is the Gold-only time. Compare to `analyze_feedback` for the same "
        "row count to see the speedup.",
        "",
        "## Notes",
        "",
        "- Lexicon backend: pure Python, no model downloads, runs offline.",
        "- HF backend: real HuggingFace models, slow on CPU. Spark "
        "  parallelises this layer; see `benchmarks/bench_spark.py`.",
        "- Peak MB is heap only (via `tracemalloc`); native allocations "
        "  by HF tokenizers and spaCy are not captured here.",
        "",
    ]
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
