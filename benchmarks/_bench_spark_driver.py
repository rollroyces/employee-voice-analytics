"""Spark driver for the bench_spark_local harness.

Spins up a SparkSession on the given master, ingests a synthetic
dataset, runs the full pipeline plus the medaillon pieces, and
prints one `MEASURE <label> <rows> <elapsed_sec>` line per stage.

Lexicon backend only — the HF backend on a single Spark local[N]
worker is dominated by driver↔executor serialisation and isn't
informative. Use `bench_hf.py` for HF CPU numbers and run HF on a
real cluster for production Spark HF numbers.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

os.environ["EMPLOYEE_VOICE_ALLOW_FALLBACK"] = "1"

import pandas as pd  # noqa: E402

import employee_voice  # noqa: E402
import employee_voice.config as _cfg  # noqa: E402
_cfg.ALLOW_FALLBACK = True

# Spark imports are deferred so the script is importable without
# pyspark installed (e.g. for unit tests).


def emit(label: str, rows: int, elapsed: float) -> None:
    """Print a single MEASURE line. The parent parses this."""
    print(f"MEASURE {label} {rows} {elapsed:.4f}")


def time_section(label: str, rows: int, fn) -> None:
    t0 = time.perf_counter()
    fn()
    emit(label, rows, time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--master", default="local[2]")
    ap.add_argument("--jar", default="io.delta:delta-spark_2.13:4.0.0",
                    help="Delta-Spark jar for the local session.")
    args = ap.parse_args()

    # Lazy Spark import so the script is importable without pyspark.
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .master(args.master)
        .appName("employee_voice_spark_bench")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", args.jar)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    # Build the synthetic dataset in pandas, then turn it into a
    # Spark DataFrame so we test the full ingestion path.
    pdf = employee_voice.generate_synthetic(n=args.rows, seed=42)
    sdf = spark.createDataFrame(pdf)

    # Register the UDFs that run_spark uses. This is normally done
    # inside analyze_spark, but we re-do it here so each stage is
    # measurable independently.
    from employee_voice import spark_udfs as _udfs
    _udfs.register_pii_udf(spark)
    _udfs.register_nlp_udfs(spark)
    _udfs.register_factor_udfs(spark)
    _udfs.register_risk_udf(spark)

    # Stage 1: full pipeline via run_spark (Bronze -> Silver -> Gold).
    from employee_voice.spark_pipeline import run_spark
    time_section("analyze_feedback_full", args.rows, lambda: run_spark(sdf))

    # Stage 2: medaillon — Silver only, then Gold-only re-score.
    from employee_voice.medallion import transform_silver, score_gold
    bronze = employee_voice.ingest_bronze(pdf)
    silver, _ = transform_silver(bronze, redact=True)
    time_section("score_gold_from_silver", args.rows, lambda: score_gold(silver))

    # Sanity: re-score is deterministic.
    gold_a, _ = score_gold(silver)
    gold_b, _ = score_gold(silver)
    assert (list(gold_a["RiskScore"].astype(float)) ==
            list(gold_b["RiskScore"].astype(float))), (
        "Gold re-score is not deterministic"
    )

    spark.stop()


if __name__ == "__main__":
    main()
