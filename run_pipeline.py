"""CLI entry point for the Employee Voice Analytics pipeline."""
from __future__ import annotations
import argparse
import logging
import os
import sys

from employee_voice.pipeline import run
from employee_voice.summarizer import generate_summary


def main() -> int:
    p = argparse.ArgumentParser(description="Employee Voice Analytics pipeline")
    p.add_argument("--input", required=True, help="Path to CSV / XLSX / JSON / Parquet")
    p.add_argument("--sheet", default=None, help="XLSX sheet name (default: first)")
    p.add_argument("--text-column", default=None,
                   help="Column with the verbatim comment (auto-detected if omitted)")
    p.add_argument("--outdir", default="output", help="Output directory")
    p.add_argument("--no-topics", action="store_true", help="Skip topic discovery")
    p.add_argument("--summary", action="store_true",
                   help="Generate LLM exec summary (requires AZURE_OPENAI_* env)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    artifacts = run(
        input_path=args.input,
        outdir=args.outdir,
        sheet=args.sheet,
        text_column=args.text_column,
        run_topics=not args.no_topics,
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
