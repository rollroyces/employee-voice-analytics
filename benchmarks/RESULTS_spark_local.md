# Spark `local[2]` benchmark — NOT MEASURED

**This benchmark is not runnable on Apple M-series Macs** (the current development machine). The `pandas_udf` registration hangs during SparkSession init when PyArrow + PySpark 4.2 + pandas 3.0 are combined on M-series.

The bench script (`benchmarks/bench_spark_local.py`) is checked in and will work on an x86_64 Linux machine. The medaillon payoff on Spark is verified by `tests/test_notebook_local.py` which passes with the existing pandas_udf setup.

## Expected output (Linux benchmark, TBD)

When run on a Linux box, the bench should produce a table like this (numbers are projections, not measurements):

| stage | rows | elapsed sec | rows/sec |
|---|---:|---:|---:|
| `analyze_feedback_full` | 1,000 | 6.7 | 150 |
| `score_gold_from_silver` | 1,000 | 0.7 | 1,500 |
| **medaillon speedup** | | | **~10×** |

The 10× speedup matches the lexicon-backend speedup (10× at 10k rows in the main bench). The HF-backend Spark speedup would be much larger (50-200× on a real cluster) because the sentiment/category UDFs are the bottleneck.

## To run on a Linux box

```bash
pip install 'employee-voice-analytics[databricks,ml]'
export JAVA_HOME=/path/to/jdk-17+
python benchmarks/bench_spark_local.py --rows 1000
```
