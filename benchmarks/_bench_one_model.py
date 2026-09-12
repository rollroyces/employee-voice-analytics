"""Per-model zero-shot classification runner. Used by
`bench_category_models.py` to time one model in isolation.

Prints a single `MEASURE <elapsed_sec>` line to stdout so the
parent can parse the timing without subprocess overhead.
"""
from __future__ import annotations
import argparse
import os
import signal
import time


def _alarm_handler(signum, frame):
    raise SystemExit(124)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HuggingFace model ID for zero-shot-classification.")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--warmup-rows", type=int, default=10)
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(args.timeout_s)

    os.environ["EMPLOYEE_VOICE_ALLOW_FALLBACK"] = "1"

    import employee_voice  # noqa: E402
    import employee_voice.config as _cfg  # noqa: E402
    _cfg.ALLOW_FALLBACK = True

    from transformers import pipeline  # noqa: E402
    pipe = pipeline("zero-shot-classification", model=args.model, top_k=1)

    df = employee_voice.generate_synthetic(n=args.rows, seed=42)
    texts = df["Comment"].fillna("").tolist()
    labels = _cfg.CATEGORIES

    # Warmup (load + first inference) — discards result.
    if args.warmup_rows > 0:
        pipe(texts[:args.warmup_rows], candidate_labels=labels, multi_label=True)

    # Measure.
    t0 = time.perf_counter()
    pipe(texts, candidate_labels=labels, multi_label=True)
    elapsed = time.perf_counter() - t0

    print(f"MEASURE {args.model} {args.rows} {elapsed:.4f}")


if __name__ == "__main__":
    main()
