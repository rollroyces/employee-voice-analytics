"""
HF-backend bench wrapper. Used to measure the HuggingFace analyzer
throughput on a single CPU. macOS doesn't ship GNU `timeout` so we
wrap the subprocess in Python's signal.alarm instead.

The HF model cold-load is ~10-20 s. The first call also has a
warmup cost. To get a clean steady-state number we run a small
warmup pass, then measure 100 rows.

Usage:
    python benchmarks/bench_hf.py --rows 100
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

# Use signal.alarm for the timeout — works on macOS, doesn't need
# GNU coreutils.
def _alarm_handler(signum, frame):
    raise TimeoutError("subprocess exceeded wall clock limit")

def run_with_timeout(cmd, timeout_s):
    """Run a subprocess and kill it if it exceeds timeout_s."""
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        return -9  # SIGKILL exit code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--out", default="benchmarks/RESULTS_hf.md")
    ap.add_argument("--csv", default="benchmarks/results_hf.csv")
    ap.add_argument("--warmup-rows", type=int, default=20)
    ap.add_argument("--timeout-s", type=int, default=1800)
    args = ap.parse_args()

    env = os.environ.copy()
    env["EMPLOYEE_VOICE_ALLOW_FALLBACK"] = "1"  # themes uses fallback
    env["PYTHONPATH"] = "."

    # Stage 1: warmup. Loads the HF models.
    print(f"=== HF warmup: {args.warmup_rows} rows ===", flush=True)
    rc = run_with_timeout(
        [
            sys.executable, "benchmarks/_hf_one_layer.py",
            "--rows", str(args.warmup_rows), "--layer", "sentiment", "--warmup",
        ],
        timeout_s=args.timeout_s,
    )
    if rc != 0:
        print(f"warmup failed (rc={rc})")
        sys.exit(rc if rc > 0 else 1)

    # Stage 2: real measurements. One layer at a time so each gets
    # its own wall clock and the first call's overhead doesn't
    # contaminate the others.
    results = []
    for layer in ["sentiment", "category", "emotion"]:
        # Re-warmup between layers because the analyzer caches the
        # pipe per process and the pipe isn't shared.
        warmup_rc = run_with_timeout(
            [
                sys.executable, "benchmarks/_hf_one_layer.py",
                "--rows", str(args.warmup_rows), "--layer", layer, "--warmup",
            ],
            timeout_s=args.timeout_s,
        )
        if warmup_rc != 0:
            print(f"warmup for {layer} failed (rc={warmup_rc})")
            continue

        # Now time the actual run.
        print(f"\n=== HF {layer}: {args.rows} rows ===", flush=True)
        t0 = time.perf_counter()
        rc = run_with_timeout(
            [
                sys.executable, "benchmarks/_hf_one_layer.py",
                "--rows", str(args.rows), "--layer", layer, "--measure",
            ],
            timeout_s=args.timeout_s,
        )
        elapsed = time.perf_counter() - t0
        if rc != 0:
            print(f"  {layer} failed (rc={rc}); skipping")
            continue
        results.append({
            "layer": layer,
            "rows": args.rows,
            "elapsed_sec": elapsed,
            "rows_per_sec": args.rows / elapsed if elapsed > 0 else float("inf"),
        })
        print(f"  {layer}: {elapsed:.2f}s = {args.rows / elapsed:.1f} rows/sec")

    if not results:
        print("no successful measurements — see above")
        sys.exit(1)

    # Write CSV.
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(args.csv, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=["layer", "rows", "elapsed_sec", "rows_per_sec"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nWrote {args.csv}")

    # Write markdown.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HuggingFace backend — single-CPU benchmark",
        "",
        f"_Generated on {platform.machine()}, Python {platform.python_version()}. "
        f"Each measurement is the steady-state time after a {args.warmup_rows}-row "
        "warmup pass. Wall-clock measured by the parent process "
        "(subprocess round-trip + HF model output parsing)._",
        "",
        "| layer | rows | elapsed sec | rows/sec |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(f"| `{r['layer']}` | {r['rows']:,} | {r['elapsed_sec']:.2f} | {r['rows_per_sec']:.1f} |")
    lines += [
        "",
        "## Reading these numbers",
        "",
        "- **CPU only**, no GPU. Apple M-series arm64. The HF models are downloaded on first use; the warmup pass pays that cost so the measurement is steady-state.",
        "- **For production**: run on a GPU cluster via `analyze_spark()`. The Spark `pandas_udf(SCALAR_ITER)` runtime parallelises per-row inference across executors; a 10x or 100x speedup over the single-CPU numbers is normal.",
        "- **The lexicon backend is ~100x faster** on the same machine, but at the cost of accuracy. Use the lexicon for tests / offline evaluation; use the HF backend in production.",
        "",
    ]
    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
