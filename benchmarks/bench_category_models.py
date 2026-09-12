"""
Category model comparison: BART-MNLI vs DeBERTa-v3-base-mnli.

The category layer is the bottleneck on a single machine:
`facebook/bart-large-mnli` does ~1.5 rows/sec because it's a
400M-parameter zero-shot model. The recommended path is to use
a smaller, distilled NLI model. This bench measures the speedup
and the accuracy trade-off.

We measure:
  1. Throughput: rows/sec for the 100-row sample on each model.
  2. Accuracy (optional, slow): how often the top-1 label matches
     the first 1-2 words of the comment (a crude but fast proxy
     for category sanity). Skipped by default; pass --accuracy
     to enable.

Models compared:
  - facebook/bart-large-mnli (~400M params, 1.5 rows/sec)
  - MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (~184M, ?)
  - MoritzLaurer/DeBERTa-v3-small-mnli-anli (~70M, ?)  [tiny option]

Usage:
    pip install 'employee-voice-analytics[ml]'
    python benchmarks/bench_category_models.py --rows 100
    python benchmarks/bench_category_models.py --rows 50 --accuracy
"""
from __future__ import annotations
import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ["EMPLOYEE_VOICE_ALLOW_FALLBACK"] = "1"
import pandas as pd

import employee_voice  # noqa: E402
import employee_voice.config as _cfg  # noqa: E402
_cfg.ALLOW_FALLBACK = True


# Models to compare. Each entry is (label, model_id).
# The smaller models (`cross-encoder/nli-deberta-v3-small`,
# `typeform/distilbert-base-uncased-mnli`) are real, small, and
# fast on CPU. DistilBERT-MNLI is the smallest of the three at
# ~70M params; the cross-encoder DeBERTa-v3-small is ~140M.
MODELS: list[tuple[str, str]] = [
    ("bart-large-mnli (400M, baseline)", "facebook/bart-large-mnli"),
    ("deberta-v3-base-mnli (140M)", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
    ("deberta-v3-small (140M)", "cross-encoder/nli-deberta-v3-small"),
    ("distilbert-base-mnli (70M)", "typeform/distilbert-base-uncased-mnli"),
]


def time_model(model_id: str, rows: int, warmup_rows: int = 10,
               timeout_s: int = 600) -> float:
    """Run the zero-shot pipeline on a given model in a subprocess
    so model download + first inference are isolated. Returns the
    layer time in seconds as reported by the child (excludes
    subprocess / interpreter startup, which the wall-clock
    measurement would over-count)."""
    cmd = [
        sys.executable, "benchmarks/_bench_one_model.py",
        "--model", model_id, "--rows", str(rows), "--warmup-rows", str(warmup_rows),
    ]
    env = os.environ.copy()
    env["EMPLOYEE_VOICE_ALLOW_FALLBACK"] = "1"
    env["PYTHONPATH"] = "."
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return float("nan")
    if proc.returncode != 0:
        print(f"  model {model_id} failed (rc={proc.returncode}):")
        print(proc.stderr[-500:])
        return float("nan")
    # Parse the MEASURE line. The child reports its own in-process
    # timing; the parent's subprocess wall-clock would include
    # Python interpreter startup + transformers import + model
    # load, which is exactly the noise we want to exclude.
    for line in proc.stdout.splitlines():
        if line.startswith("MEASURE"):
            parts = line.split()
            try:
                # parts: MEASURE <model> <rows> <elapsed_sec>
                return float(parts[3])
            except (ValueError, IndexError):
                continue
    print(f"  model {model_id}: no MEASURE line in stdout")
    return float("nan")


def accuracy_proxy(model_id: str, df: pd.DataFrame) -> float:
    """For each row, check whether the model's top-1 category label
    appears as a substring of the comment. Crude but fast.

    Returns the fraction of rows where the top-1 label (case-folded,
    first word) appears anywhere in the comment. Higher = more
    sensible output.
    """
    from transformers import pipeline
    pipe = pipeline("zero-shot-classification", model=model_id, top_k=1)
    labels = employee_voice.config.CATEGORIES

    hits = 0
    for _, row in df.iterrows():
        text = str(row.get("Comment", ""))[:512]
        if not text.strip():
            continue
        try:
            out = pipe(text, candidate_labels=labels, multi_label=False)
            top_label = out["labels"][0].lower()
            # Take the first word of the label (e.g. "Compensation and
            # Benefits" -> "compensation") and check for a substring.
            first_word = top_label.split()[0]
            if first_word in text.lower():
                hits += 1
        except Exception:
            continue
    return hits / max(1, len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--models", default=",".join(m[1] for m in MODELS),
                    help="Comma-separated model IDs to compare.")
    ap.add_argument("--out", default="benchmarks/RESULTS_category_models.md")
    ap.add_argument("--csv", default="benchmarks/results_category_models.csv")
    ap.add_argument("--accuracy", action="store_true",
                    help="Run a slow accuracy proxy in addition to throughput.")
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()

    print(f"=== Category model bench: rows={args.rows} models={args.models} ===\n",
          flush=True)

    # Synthetic data — same distribution as the main bench.
    df = employee_voice.generate_synthetic(n=args.rows, seed=42)

    results = []
    for label, model_id in [(m[0], m[1]) for m in MODELS if m[1] in args.models.split(",")]:
        # Skip if we only want specific labels.
        if model_id not in args.models.split(","):
            continue
        print(f"-- {label} ({model_id}) --", flush=True)
        elapsed = time_model(model_id, args.rows, timeout_s=args.timeout_s)
        rps = args.rows / elapsed if elapsed > 0 else float("inf")
        acc = None
        if args.accuracy:
            print("   computing accuracy proxy...", flush=True)
            acc = accuracy_proxy(model_id, df.head(50))  # cap at 50 for speed
        results.append({
            "label": label,
            "model": model_id,
            "rows": args.rows,
            "elapsed_sec": elapsed,
            "rows_per_sec": rps,
            "accuracy_proxy": acc if acc is not None else "",
        })
        print(f"   {label}: {elapsed:.2f}s = {rps:.1f} rows/sec"
              + (f" (acc≈{acc:.0%})" if acc is not None else ""))

    if not results:
        print("no successful measurements")
        sys.exit(1)

    # CSV.
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.csv, index=False)
    print(f"\nWrote {args.csv}")

    # Markdown.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Category model comparison — single-machine benchmark",
        "",
        f"_Generated on {platform.machine()}, Python {platform.python_version()}, "
        f"{args.rows} rows._",
        "",
        "Category is the bottleneck in the HF backend (see "
        "`benchmarks/RESULTS_hf.md`). This bench measures whether a "
        "smaller zero-shot NLI model recovers most of the throughput "
        "without giving up too much accuracy.",
        "",
        "| model | rows | elapsed sec | rows/sec | accuracy proxy |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        acc_cell = f"{r['accuracy_proxy']:.0%}" if r["accuracy_proxy"] != "" else "n/a"
        lines.append(
            f"| `{r['label']}` ({r['model']}) | {r['rows']:,} | "
            f"{r['elapsed_sec']:.2f} | {r['rows_per_sec']:.1f} | {acc_cell} |"
        )
    lines += [
        "",
        "## Reading the accuracy proxy",
        "",
        "The accuracy column is a *very* crude sanity check: it returns 1.0 "
        "if the top-1 model's first word appears as a substring of the "
        "comment. So for a comment like 'Compensation is below market', "
        "the model needs to predict any label starting with 'compensation' "
        "(e.g. 'Compensation and Benefits') to score 1.0. This is much "
        "weaker than a real label-quality benchmark on a held-out labelled "
        "set, but it's enough to spot a model that's completely lost. "
        "Use the [`setfit` library](https://github.com/huggingface/setfit) "
        "on a few hundred labelled comments for a real accuracy measurement.",
        "",
    ]
    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
