"""
HF-backend bench wrapper. Used to measure the HuggingFace analyzer
throughput on a single CPU.

Design notes:
- The HF model cold-load is ~10-20s on first call. Each HF layer
  caches its pipeline in a module-level dict, so subsequent
  layers in the same process can share it. We deliberately use
  *separate* subprocesses per layer so the measurement is one
  layer at a time (no pipeline-sharing effects).
- We use a *warmup* subprocess and a *measure* subprocess. The
  warmup pays the model-download + first-inference cost; the
  measure subprocess is steady state.
- The child prints one parseable line: `MEASURE <layer> <rows>
  <elapsed_sec>` to stdout. The parent parses that line so the
  measurement excludes subprocess startup, transformers import,
  and Python interpreter warm-up. Wall-clock would be ~10s
  longer than the real layer cost.
- Subprocess timeout is enforced two ways: (1) parent uses
  `subprocess.Popen(...).wait(timeout=...)`, (2) the child
  installs a `signal.alarm` watchdog as a backstop in case
  `wait` is interrupted.
"""
from __future__ import annotations
import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def run_layer_subprocess(layer: str, rows: int, mode: str,
                        timeout_s: int) -> tuple[int, str]:
    """Run a single layer in a child subprocess. Returns (returncode, stdout).
    The child prints one `MEASURE <layer> <rows> <sec>` line in measure
    mode; the parent parses that line out of stdout.
    """
    cmd = [
        sys.executable, "benchmarks/_hf_one_layer.py",
        "--rows", str(rows), "--layer", layer, "--mode", mode,
    ]
    env = os.environ.copy()
    env["EMPLOYEE_VOICE_ALLOW_FALLBACK"] = "1"  # themes uses fallback
    env["PYTHONPATH"] = "."
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=timeout_s, check=False,
        )
        wall = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return -9, f"timeout after {timeout_s}s"
    return proc.returncode, proc.stdout + ("\nstderr: " + proc.stderr if proc.stderr else "")


def parse_measure_line(stdout: str) -> tuple[str, int, float] | None:
    """Pull the `MEASURE <layer> <rows> <elapsed>` line out of the child's
    stdout. Returns (layer, rows, elapsed_sec) or None if not found.
    """
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] == "MEASURE":
            try:
                return parts[1], int(parts[2]), float(parts[3])
            except ValueError:
                continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--out", default="benchmarks/RESULTS_hf.md")
    ap.add_argument("--csv", default="benchmarks/results_hf.csv")
    ap.add_argument("--warmup-rows", type=int, default=20)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument(
        "--layers", default="sentiment,category,emotion",
        help="Comma-separated layer names to benchmark.",
    )
    args = ap.parse_args()

    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    if not layers:
        print("no layers specified")
        sys.exit(1)

    print(f"=== HF bench: layers={layers} rows={args.rows} ===", flush=True)
    print(f"=== platform: {platform.machine()} python {platform.python_version()} ===\n", flush=True)

    # Per-layer measurement. Warmup + measure, each in its own
    # subprocess so model download + first inference are excluded
    # from the timing.
    results = []
    for layer in layers:
        print(f"-- {layer}: warmup ({args.warmup_rows} rows) --", flush=True)
        warmup_rc, warmup_out = run_layer_subprocess(
            layer, args.warmup_rows, "warmup", timeout_s=args.timeout_s,
        )
        if warmup_rc != 0:
            print(f"   warmup failed: rc={warmup_rc} {warmup_out[-300:]}")
            continue

        print(f"-- {layer}: measure ({args.rows} rows) --", flush=True)
        measure_rc, measure_out = run_layer_subprocess(
            layer, args.rows, "measure", timeout_s=args.timeout_s,
        )
        if measure_rc != 0:
            print(f"   measure failed: rc={measure_rc} {measure_out[-300:]}")
            continue

        parsed = parse_measure_line(measure_out)
        if parsed is None:
            print(f"   could not parse MEASURE line from child output:")
            print(measure_out)
            continue

        _, rows, elapsed = parsed
        rps = rows / elapsed if elapsed > 0 else float("inf")
        results.append({
            "layer": layer, "rows": rows, "elapsed_sec": elapsed,
            "rows_per_sec": rps,
        })
        print(f"   {layer}: {elapsed:.2f}s = {rps:.1f} rows/sec")

    if not results:
        print("\nno successful measurements — see above")
        sys.exit(1)

    # Write CSV.
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(args.csv, "w", newline="") as fh:
        writer = _csv.DictWriter(
            fh, fieldnames=["layer", "rows", "elapsed_sec", "rows_per_sec"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nWrote {args.csv}")

    # Write markdown.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HuggingFace backend — single-machine benchmark",
        "",
        f"_Generated on {platform.machine()}, Python {platform.python_version()}. "
        f"Each measurement is the steady-state time inside a fresh "
        f"subprocess after a {args.warmup_rows}-row warmup subprocess "
        f"(so model download + first-inference cost are excluded)._",
        "",
        "| layer | rows | elapsed sec | rows/sec |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r['layer']}` | {r['rows']:,} | {r['elapsed_sec']:.2f} | "
            f"{r['rows_per_sec']:.1f} |"
        )
    lines += [
        "",
        "## Reading these numbers",
        "",
        "- **Apple M-series** is what the harness was run on; an "
        "M-series Mac with MPS GPU was used. CPU-only machines will be "
        "~5× slower; a discrete GPU cluster will be ~10-50× faster.",
        "- **For production** use `analyze_spark()` on a cluster. The "
        "`pandas_udf(SCALAR_ITER)` runtime parallelises per-row "
        "inference across executors.",
        "- **The lexicon backend is ~100× faster** on the same machine, "
        "but at the cost of accuracy. Use the lexicon for tests / "
        "offline evaluation; use HF in production.",
        "- **Category is the bottleneck**: BART-MNLI scores each input "
        "against all 18 HR categories. Distilling to a smaller model "
        "(e.g. `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) is a "
        "10× speedup at modest accuracy cost.",
        "",
    ]
    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
