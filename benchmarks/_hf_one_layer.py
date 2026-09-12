"""Per-layer HF runner. Used by `bench_hf.py` to measure one layer at
a time. Prints a single `MEASURE <layer> <rows> <elapsed_sec>` line
to stdout so the parent process can parse the timing without
subprocess / interpreter overhead.

Usage:
    python benchmarks/_hf_one_layer.py --rows 100 --layer sentiment --mode warmup
    python benchmarks/_hf_one_layer.py --rows 100 --layer sentiment --mode measure
"""
from __future__ import annotations
import argparse
import os
import signal
import time


# A watchdog for the child. If the parent process loses its
# connection to us (e.g. SIGKILL on parent) we'd otherwise hang
# forever on a slow HF forward pass. The signal handler turns
# the alarm into a clean exit.
def _alarm_handler(signum, frame):
    raise SystemExit(124)  # same convention as GNU `timeout`


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--layer", choices=["sentiment", "category", "emotion"],
                    required=True)
    ap.add_argument("--mode", choices=["warmup", "measure"], required=True)
    ap.add_argument("--timeout-s", type=int, default=1500,
                    help="Child watchdog: SIGALRM after this many seconds.")
    args = ap.parse_args()

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(args.timeout_s)

    # Force HF for the chosen layer. The analyzer's _current_backend
    # is read at call time, so toggling ALLOW_FALLBACK + overriding
    # the per-layer _BACKENDS cache is enough.
    os.environ["EMPLOYEE_VOICE_ALLOW_FALLBACK"] = "1"  # themes exempt

    import employee_voice  # noqa: E402
    import employee_voice.config as _cfg  # noqa: E402
    _cfg.ALLOW_FALLBACK = True
    from employee_voice.analyzers import (  # noqa: E402
        analyze_sentiment, analyze_category, analyze_emotion, _Backends,
    )
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

    if args.mode == "measure":
        # Single parseable line. The parent reads stdout and pulls
        # this out; the rest of stdout is for human debugging.
        print(f"MEASURE {args.layer} {args.rows} {elapsed:.4f}")


if __name__ == "__main__":
    main()
