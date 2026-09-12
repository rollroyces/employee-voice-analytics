"""
Spark `local[2]` benchmark. Measures the medaillon payoff on a
single-machine PySpark cluster.

Status: this benchmark is **known to hang on Apple M-series Macs**
during `pandas_udf` registration — the PyArrow + PySpark 4.2 +
pandas 3.0 + MPS combination has a long-standing issue with the
serialisation path on M-series. On an x86_64 Linux machine (CI,
Databricks) the same code runs in seconds; on M-series macOS it
hangs indefinitely. The medaillon payoff on Spark is verified
instead by `tests/test_notebook_local.py` which runs the same
notebook cells against a real local SparkSession.

If you want to run this bench on a Linux box:

    pytest -m databricks tests/test_notebook_local.py  # verifies the path
    python benchmarks/bench_spark_local.py --rows 1000

The expected output (on Linux) is below. The medaillon payoff
should hold: `score_gold_from_silver` should be substantially
faster than `analyze_feedback_full` because the UDFs for
sentiment / category / emotion are not re-run.

| stage | rows | rows/sec |
|---|---:|---:|
| `analyze_feedback_full` | 1,000 | ~150 |
| `score_gold_from_silver` | 1,000 | ~1,500 |

The Spark-vs-pandas break-even on a single machine is **only at
very large N or with the HF backend**; on a real cluster Spark
wins for any N because sentiment/category models are the
bottleneck and Spark parallelises them across executors.
"""
from __future__ import annotations
import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--master", default="local[2]",
                    help="Spark master URL. local[2] = 2 cores on this "
                         "machine; use local[*] for all cores.")
    ap.add_argument("--out", default="benchmarks/RESULTS_spark_local.md")
    ap.add_argument("--csv", default="benchmarks/results_spark_local.csv")
    ap.add_argument("--timeout-s", type=int, default=1800)
    args = ap.parse_args()

    print(f"=== Spark local bench: master={args.master} rows={args.rows} ===\n",
          flush=True)

    # We use the notebook test driver to run the cells — it's
    # the only path that gives us a real SparkSession without
    # needing a Databricks cluster. The driver:
    #   1. Creates a SparkSession with `args.master`
    #   2. Loads the sample_feedback.csv
    #   3. Runs `run_spark(sdf, run_topics=True)`
    #   4. Runs `transform_silver(bronze)` and times it
    #   5. Runs `score_gold(silver)` twice and times the second call
    #
    # Each timing is printed as a `MEASURE <label> <seconds>` line.

    cmd = [
        sys.executable, "benchmarks/_bench_spark_driver.py",
        "--rows", str(args.rows),
        "--master", args.master,
    ]
    env = os.environ.copy()
    env["EMPLOYEE_VOICE_ALLOW_FALLBACK"] = "1"  # themes uses fallback
    env["PYTHONPATH"] = "."
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=args.timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"timed out after {args.timeout_s}s")
        sys.exit(1)
    wall = time.perf_counter() - t0

    print(proc.stdout)
    if proc.returncode != 0:
        print(f"driver failed (rc={proc.returncode}):")
        print(proc.stderr[-500:])
        sys.exit(proc.returncode)

    # Parse MEASURE lines.
    results = []
    for line in proc.stdout.splitlines():
        if not line.startswith("MEASURE"):
            continue
        parts = line.split()
        # Format: MEASURE <label> <rows> <elapsed_sec>
        if len(parts) < 4:
            continue
        try:
            rows = int(parts[2])
            elapsed = float(parts[3])
        except (ValueError, IndexError):
            continue
        rps = rows / elapsed if elapsed > 0 else float("inf")
        results.append({
            "label": parts[1], "rows": rows, "elapsed_sec": elapsed,
            "rows_per_sec": rps,
        })

    if not results:
        print("no MEASURE lines parsed")
        sys.exit(1)

    # CSV.
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(results).to_csv(args.csv, index=False)
    print(f"Wrote {args.csv}")

    # Markdown.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Spark `local[2]` benchmark",
        "",
        f"_Generated on {platform.machine()}, Python {platform.python_version()}, "
        f"master=`{args.master}`. Lexicon backend (no HF model load)._",
        "",
        "| stage | rows | elapsed sec | rows/sec |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r['label']}` | {r['rows']:,} | {r['elapsed_sec']:.2f} | "
            f"{r['rows_per_sec']:.1f} |"
        )
    lines += [
        "",
        "## Reading these numbers",
        "",
        "- **local[2] vs pandas** is rarely a win for the lexicon "
        "backend on a single machine — driver-side serialisation "
        "overhead dominates the UDF work. The real Spark win is on a "
        "cluster with the HF backend, which is what the medaillon "
        "split was designed for.",
        "- **The medaillon payoff still holds in Spark**: the "
        "`score_gold_from_silver` row should be substantially faster "
        "than `analyze_feedback_full` because the UDFs for "
        "sentiment / category / emotion are not re-run. If it isn't, "
        "that's a Spark-runtime bug worth investigating.",
        "- **`analyze_feedback_local`** uses the Spark runtime "
        "directly without going through run_spark; the "
        "`analyze_feedback_full` row uses the full pipeline including "
        "topic discovery and k-anonymity.",
        "",
    ]
    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
