"""Per-layer HF runner. Used by `bench_hf.py` to measure one layer at
a time so cold-load and warmup costs are isolated.

Usage:
    python benchmarks/_hf_one_layer.py --rows 100 --layer sentiment --warmup
    python benchmarks/_hf_one_layer.py --rows 100 --layer sentiment --measure
"""
from __future__ import annotations
import argparse
import os
import sys
import time

# Force HF backend (lexicon is for tests only).
os.environ["EMPLOYEE_VOICE_ALLOW_FALLBACK"] = "1"  # themes exempt

# These imports happen after the env var is set so analyzers._detect_backends
# sees the right flag.
import employee_voice  # noqa: E402
import employee_voice.config as _cfg  # noqa: E402
_cfg.ALLOW_FALLBACK = True  # themes uses fallback
from employee_voice.analyzers import (  # noqa: E402
    analyze_sentiment, analyze_category, analyze_emotion,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--layer", choices=["sentiment", "category", "emotion"],
                    required=True)
    ap.add_argument("--warmup", action="store_true",
                    help="Run a throwaway pass; the result is discarded.")
    ap.add_argument("--measure", action="store_true",
                    help="Run the actual measurement; print timing to stdout.")
    args = ap.parse_args()

    if not (args.warmup or args.measure):
        ap.error("specify --warmup or --measure")

    # Force HF for the chosen layer. We do this AFTER the analyzer
    # module has been imported (it captures ALLOW_FALLBACK at call
    # time, not at import time, so this is safe).
    from employee_voice.analyzers import _Backends
    import employee_voice.analyzers as _a
    _a._BACKENDS = _Backends(
        sentiment="transformers",
        category="transformers",
        emotion="transformers",
    )

    df = employee_voice.generate_synthetic(n=args.rows, seed=42)
    texts = df["Comment"].fillna("").tolist()

    t0 = time.perf_counter()
    if args.layer == "sentiment":
        analyze_sentiment(texts)
    elif args.layer == "category":
        analyze_category(texts)
    else:
        analyze_emotion(texts)
    elapsed = time.perf_counter() - t0

    if args.measure:
        # Print a single number the parent process can parse.
        print(f"MEASURE {args.layer} {args.rows} {elapsed:.4f}")


if __name__ == "__main__":
    main()
