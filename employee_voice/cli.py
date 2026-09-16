"""Command-line entry point for the Employee Voice Analytics pipeline.

Installed as a console script `employee-voice` when the package is on PYTHONPATH.
The legacy `run_pipeline.py` at the repo root is a thin shim that re-exports
`main()` for backwards compatibility.
"""
from __future__ import annotations
import argparse
import logging
import os
import sys

from employee_voice.pipeline import run
from employee_voice.summarizer import generate_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="employee-voice",
        description="Employee Voice Analytics pipeline (sentiment + category + "
                    "emotion + themes + attrition risk).",
    )
    p.add_argument("--input", required=True, help="Path to CSV / XLSX / JSON / Parquet")
    p.add_argument("--sheet", default=None, help="XLSX sheet name (default: first)")
    p.add_argument("--text-column", default=None,
                   help="Column with the verbatim comment (auto-detected if omitted)")
    p.add_argument("--outdir", default="output", help="Output directory")
    p.add_argument("--no-topics", action="store_true", help="Skip topic discovery")
    p.add_argument("--summary", action="store_true",
                   help="Generate LLM exec summary (requires AZURE_OPENAI_* env)")
    p.add_argument("--verbose", "-v", action="store_true")
    # --- Hosted-LLM backend (opt-in). Default: local HF / fallback. ---
    p.add_argument("--sentiment-backend", choices=["auto", "hf", "fallback", "llm"],
                   default="auto",
                   help="Sentiment layer backend. 'llm' requires AZURE_OPENAI_* "
                        "or OPENAI_API_KEY or ANTHROPIC_API_KEY in env.")
    p.add_argument("--category-backend", choices=["auto", "hf", "fallback", "llm"],
                   default="auto",
                   help="Category layer backend. 'llm' requires an LLM provider.")
    p.add_argument("--emotion-backend", choices=["auto", "hf", "fallback", "llm"],
                   default="auto",
                   help="Emotion layer backend. 'llm' requires an LLM provider.")
    p.add_argument("--llm-batch-size", type=int, default=50,
                   help="Comments per LLM call (default 50; tune for context window).")
    p.add_argument("--llm-max-concurrent", type=int, default=1,
                   help="Parallel LLM HTTP requests (default 1 = serial).")
    p.add_argument("--send-raw-text", action="store_true",
                   help="Bypass the auto PII-scrubber before LLM calls. "
                        "Strongly discouraged for HR data; prints a warning.")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX exit code."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    # Warn loudly if the user is opting into raw-text LLM sends.
    if args.send_raw_text and any(
        b == "llm" for b in (args.sentiment_backend, args.category_backend, args.emotion_backend)
    ):
        print(
            "\n!!! WARNING: --send-raw-text is set with one or more LLM backends.\n"
            "    Comment text WILL be sent to a third-party API without PII scrubbing.\n"
            "    This is unsafe for HR data unless you have a BAA / data-residency agreement.\n",
            file=sys.stderr,
        )

    # Warn if any LLM backend is selected but no provider is configured.
    if any(b == "llm" for b in (args.sentiment_backend, args.category_backend, args.emotion_backend)):
        from employee_voice.llm_backend import detect_provider
        if detect_provider() is None:
            print(
                "\n!!! ERROR: LLM backend selected but no provider is configured.\n"
                "    Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY, or\n"
                "    OPENAI_API_KEY, or ANTHROPIC_API_KEY in the environment.\n",
                file=sys.stderr,
            )
            return 3

    llm_options = {
        "batch_size": args.llm_batch_size,
        "max_concurrent": args.llm_max_concurrent,
        "send_raw_text": args.send_raw_text,
    }

    artifacts = run(
        input_path=args.input,
        outdir=args.outdir,
        sheet=args.sheet,
        text_column=args.text_column,
        run_topics=not args.no_topics,
        sentiment_backend=args.sentiment_backend,
        category_backend=args.category_backend,
        emotion_backend=args.emotion_backend,
        llm_options=llm_options,
    )

    print("\n=== Pipeline complete ===")
    print(f"  fact:      {artifacts['fact_path']}")
    print(f"  dim_topic: {artifacts['dim_topic_path']}")
    print(f"  summary:   {artifacts['summary_path']}")
    if artifacts.get("bu_path"):
        print(f"  bu rollup: {artifacts['bu_path']}")

    fact = artifacts["fact"]
    print(f"\n  rows:           {len(fact)}")
    print(f"  analyzable:     {int((fact['Sentiment'] != 'No Comment').sum())}")
    if "RiskBand" in fact.columns:
        for band in ("High", "Medium", "Low"):
            n = int((fact["RiskBand"] == band).sum())
            print(f"  risk = {band:6s}: {n}")

    if args.summary:
        out = generate_summary(fact, args.outdir)
        if out:
            print(f"  exec summary: {out}")
        else:
            print("  exec summary: skipped (Azure OpenAI not configured)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
