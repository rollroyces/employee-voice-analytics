"""Drive databricks_notebook.py as plain Python to verify it runs end-to-end
on a local Spark session. NOT a substitute for running on real Databricks,
but exercises every cell, catches import / syntax / API errors, and confirms
the local pipeline + Spark compose correctly.

Run with:
    export JAVA_HOME=$HOME/.local/jdk/jdk-21.0.12.1.jdk/Contents/Home
    export PATH=$JAVA_HOME/bin:$PATH
    pip install 'employee-voice-analytics[databricks]'
    pytest -m databricks tests/test_notebook_local.py

The notebook still ships the production Delta paths; this driver rewrites
dbfs:/ URIs to local paths and Delta -> Parquet for the test run only.

Marked @pytest.mark.databricks so plain `pytest` skips it on machines without
Java + PySpark + Delta installed.
"""
from __future__ import annotations
import os
import re
import sys

import pytest

# Make the package importable regardless of cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

NB_PATH = os.path.join(PROJECT, "databricks_notebook.py")


# -- Minimal dbutils shim -------------------------------------------------
class _Widgets:
    def __init__(self):
        self._values = {
            "input_path": os.path.join(PROJECT, "data", "sample_feedback.csv"),
            "output_catalog": "",
            "output_database": "default",
            "text_column": "",
            "sheet": "",
            "run_topics": "true",
            "run_summary": "false",
        }
    def text(self, name, default, _label=None): pass
    def dropdown(self, name, default, _choices, _label): pass
    def get(self, name):
        return self._values.get(name, "")

class _Secrets:
    def get(self, *_args, **_kw): raise RuntimeError("no secrets in local mode")

class _DBUtils:
    def __init__(self):
        self.widgets = _Widgets()
        self.secrets = _Secrets()


# -- Cell splitter --------------------------------------------------------
def split_cells(path):
    text = open(path).read()
    raw = re.split(r"^\s*# COMMAND\s*-{3,}\s*$", text, flags=re.M)
    return [c.strip() for c in raw if c.strip()]


# -- Execute --------------------------------------------------------------
def _run_notebook() -> None:
    """Execute every cell in the notebook against a real local SparkSession."""
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("employee_voice_notebook_test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        # Local-only: register Delta (Databricks already has it on the classpath).
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-spark_2.13:4.0.0")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    local_out = os.path.join(PROJECT, "_test_output")
    if os.path.exists(local_out):
        import shutil
        shutil.rmtree(local_out)
    os.makedirs(local_out, exist_ok=True)

    g = {
        "__name__": "__main__",
        "spark": spark,
        "dbutils": _DBUtils(),
        "sc": spark.sparkContext,
    }
    g["INPUT_PATH"] = os.path.join(PROJECT, "data", "sample_feedback.csv")

    cells = split_cells(NB_PATH)
    rewrite_map = {
        "dbfs:/FileStore/employee_voice/output": local_out,
        "dbfs:/FileStore/employee_voice/sample_feedback.csv":
            os.path.join(PROJECT, "data", "sample_feedback.csv"),
        "/dbfs": local_out,
    }
    rewrite_map_local_only = [
        ('.format("delta")', '.format("parquet")'),
    ]

    def rewrite(text: str) -> str:
        for src, dst in rewrite_map.items():
            text = text.replace(src, dst)
        for src, dst in rewrite_map_local_only:
            text = text.replace(src, dst)
        text = re.sub(
            r"\(\s*[a-z_]+\.write[^)]*\.saveAsTable\([^)]+\)\s*\)",
            "# saveAsTable skipped locally (no catalog); see databricks_notebook.py for the real call",
            text,
        )
        return text

    failures: list[tuple[int, str]] = []
    for i, cell in enumerate(cells, 1):
        lines = [ln for ln in cell.splitlines()
                 if not ln.lstrip().startswith("%")]
        cleaned = "\n".join(lines)
        cleaned = rewrite(cleaned)
        if not cleaned.strip():
            continue
        try:
            exec(compile(cleaned, f"<cell {i}>", "exec"), g)
        except Exception as exc:  # noqa: BLE001
            failures.append((i, f"{type(exc).__name__}: {exc}"))

    spark.stop()

    if failures:
        details = "\n".join(f"  cell {i}: {msg}" for i, msg in failures)
        pytest.fail(f"Notebook execution failed:\n{details}")


@pytest.mark.databricks
def test_notebook_runs_locally():
    _run_notebook()
