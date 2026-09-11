# Databricks / PySpark version of the Employee Voice Analytics pipeline.
#
# Drop this file into your workspace as a notebook (Databricks detects
# "# COMMAND ----------" cell separators), or run as a Python script with
# `spark-submit` / `%run` from a notebook.
#
# Reuses the local `employee_voice/` package — analyzer backend auto-detects:
#   - if `transformers` is installed on the cluster, uses HF pipelines
#   - otherwise falls back to the keyword + TF-IDF/KMeans engines
# Output columns are identical to the local pipeline, so the same Power BI
# model works against either source.
#
# Widget parameters (set in the Job cluster UI or via %run / dbutils.widgets):
#   input_path      (default: dbfs:/FileStore/employee_voice/sample_feedback.csv)
#   output_catalog  (default: "" — if empty, writes Delta to /output/ in DBFS)
#   output_database (default: "" — if empty, uses output_catalog.schema = "default")
#   text_column     (default: auto-detect)
#   sheet           (only for .xlsx; default: first sheet)
#   run_topics      (default: true)
#   run_summary     (default: false — requires AZURE_OPENAI_* secrets in scope)

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt 2>/dev/null || true
#
# NOTE: the heavy ML deps (transformers, bertopic, openai) are commented out
# in requirements.txt. Uncomment them in the cluster's library config or in
# this cell when you want the GPU / zero-shot path. The notebook runs end-to-end
# with just pandas + scikit-learn + openpyxl via the fallback engines.

# COMMAND ----------

import os
import sys
import logging
from typing import Optional

# Package import — when running on Databricks via `%pip install -e .` or
# by attaching the wheel, `employee_voice` is already on PYTHONPATH. The
# fallback below covers the case where the notebook sits next to the
# package in a workspace folder mounted to /Workspace/Users/.../repo.
try:
    from employee_voice import config as cfg  # noqa: F401
    from employee_voice.layers import run_all_layers  # noqa: F401
except ImportError:
    _here = os.path.dirname(os.path.abspath("databricks_notebook.py"))
    _parent = os.path.dirname(_here)
    for p in (_here, _parent):
        if p not in sys.path:
            sys.path.insert(0, p)
    from employee_voice import config as cfg  # noqa: E402,F401
    from employee_voice.layers import run_all_layers  # noqa: E402,F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("employee_voice.databricks")

# COMMAND ----------

# Widgets — visible in the Databricks Job UI; settable via dbutils.widgets.text
try:
    dbutils  # type: ignore[name-defined]
except NameError:
    dbutils = None  # running outside Databricks (e.g. spark-submit)

if dbutils is not None:
    dbutils.widgets.text("input_path", "dbfs:/FileStore/employee_voice/sample_feedback.csv", "Input path")
    dbutils.widgets.text("output_catalog", "", "Unity Catalog name (optional)")
    dbutils.widgets.text("output_database", "default", "Output schema")
    dbutils.widgets.text("text_column", "", "Comment column (blank = auto)")
    dbutils.widgets.text("sheet", "", "XLSX sheet (blank = first)")
    dbutils.widgets.dropdown("run_topics", "true", ["true", "false"], "Run topic discovery")
    dbutils.widgets.dropdown("run_summary", "false", ["true", "false"], "LLM exec summary")

def _w(name: str, default: str = "") -> str:
    if dbutils is None:
        return default
    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default

INPUT_PATH = _w("input_path", "dbfs:/FileStore/employee_voice/sample_feedback.csv")
OUTPUT_CATALOG = _w("output_catalog", "")
OUTPUT_DATABASE = _w("output_database", "default")
TEXT_COLUMN = _w("text_column", "") or None
SHEET = _w("sheet", "") or None
RUN_TOPICS = _w("run_topics", "true").lower() == "true"
RUN_SUMMARY = _w("run_summary", "false").lower() == "true"

print(f"INPUT_PATH        = {INPUT_PATH}")
print(f"OUTPUT_CATALOG    = {OUTPUT_CATALOG or '(none)'}")
print(f"OUTPUT_DATABASE   = {OUTPUT_DATABASE}")
print(f"TEXT_COLUMN       = {TEXT_COLUMN or '(auto)'}")
print(f"SHEET             = {SHEET or '(first)'}")
print(f"RUN_TOPICS        = {RUN_TOPICS}")
print(f"RUN_SUMMARY       = {RUN_SUMMARY}")

# COMMAND ----------

# Read input.
# Spark can natively read CSV / JSON / Parquet / Delta. For XLSX we drop to
# pandas (openpyxl) and rewrap into a Spark DataFrame — XLSX is rare for
# bulk HR data, but worth supporting for one-off analyst exports.
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructType, StructField, DoubleType, IntegerType

ext = os.path.splitext(INPUT_PATH)[1].lower()

if ext in (".csv", ".tsv", ".txt"):
    sep = "\t" if ext == ".tsv" else ","
    sdf = (spark.read
           .option("header", "true")
           .option("inferSchema", "true")
           .option("multiLine", "true")
           .option("escape", '"')
           .csv(INPUT_PATH, sep=sep))
elif ext == ".json":
    sdf = spark.read.option("multiLine", "true").json(INPUT_PATH)
elif ext == ".parquet":
    sdf = spark.read.parquet(INPUT_PATH)
elif ext in (".xlsx", ".xls"):
    # XLSX is unsupported natively by Spark — pull through pandas and rewrap.
    local_path = INPUT_PATH.replace("dbfs:", "/dbfs")
    pdf = pd.read_excel(local_path, sheet_name=SHEET or 0)
    sdf = spark.createDataFrame(pdf)
else:
    # Delta / unknown — let Spark figure it out.
    sdf = spark.read.format("delta").load(INPUT_PATH)

log.info("Loaded %d rows from %s", sdf.count(), INPUT_PATH)
sdf.printSchema()

# COMMAND ----------

# Normalise column names — same rules as the local pipeline.
ALIAS_MAP = {
    "feedback_id": "FeedbackID", "id": "FeedbackID", "responseid": "FeedbackID",
    "survey_type": "SurveyType", "surveyname": "SurveyType",
    "survey_date": "SurveyDate", "date": "SurveyDate", "completed_at": "SurveyDate",
    "bu": "BU", "businessunit": "BU", "business": "BU",
    "department": "Dept", "team": "Dept",
    "employee_group": "EmployeeGroup", "group": "EmployeeGroup", "tenuregroup": "EmployeeGroup",
}

def _norm(c: str) -> str:
    return c.strip().replace(" ", "").replace("_", "").lower()

def _canon(df):
    rename = {}
    fields = df.schema.fieldNames() if callable(df.schema.fieldNames) else df.schema.fieldNames
    for f in fields:
        canon = _norm(f)
        rename[f] = ALIAS_MAP.get(canon, f)
    for old, new in rename.items():
        if old != new:
            df = df.withColumnRenamed(old, new)
    return df

sdf = _canon(sdf)

# Pick / rename the comment column.
if TEXT_COLUMN:
    if TEXT_COLUMN not in sdf.columns:
        # case-insensitive match
        match = next((c for c in sdf.columns if c.lower() == TEXT_COLUMN.lower()), None)
        if not match:
            raise ValueError(f"--text-column '{TEXT_COLUMN}' not found. Available: {sdf.columns}")
        TEXT_COLUMN = match
    if TEXT_COLUMN != "Comment":
        sdf = sdf.withColumnRenamed(TEXT_COLUMN, "Comment")
else:
    for cand in ("Comment", "Comments", "Feedback", "Response", "Verbatim", "Text", "Answer"):
        if cand in sdf.columns:
            if cand != "Comment":
                sdf = sdf.withColumnRenamed(cand, "Comment")
            break
    else:
        raise ValueError(f"No comment column found. Pass --text-column. Available: {sdf.columns}")

# Ensure FeedbackID exists.
if "FeedbackID" not in sdf.columns:
    sdf = sdf.withColumn(
        "FeedbackID",
        F.concat(F.lit("F"), F.lpad(F.monotonically_increasing_id().cast("string"), 6, "0")),
    )

# Coerce date column to date type if present and stringly-typed.
if "SurveyDate" in sdf.columns:
    sdf = sdf.withColumn("SurveyDate", F.to_date("SurveyDate"))

print("Schema after normalisation:")
sdf.printSchema()
print("Row count:", sdf.count())

# COMMAND ----------

# Apply preprocessing in pandas, then re-attach to Spark.
#
# Why not a Python UDF? `clean_text` / `is_non_answer` are trivial, but the
# analyzer stack (sentiment / category / emotion) holds models / tokenizers
# that should not be pickled and broadcast to every executor. We instead
# collect to the driver once, run the analyzers there, and write the
# enriched Spark DataFrame back. For >1M comments, refactor to a Pandas UDF
# that loads HF pipelines lazily inside the function — see TODO at the end.

_PASSTHROUGH = ["FeedbackID", "SurveyType", "SurveyDate", "BU", "Dept", "EmployeeGroup", "Comment"]

pdf = sdf.select(*[c for c in _PASSTHROUGH if c in sdf.columns]).toPandas()
pdf["Comment"] = pdf["Comment"].fillna("").astype(str)

# Run Layers 0-5 under the same contract as the local pipeline. Raises
# LayerDependencyError on missing deps and LayerContractError on schema
# violations. The notebook's RUN_TOPICS widget selects whether Layer 4
# is in scope; Layer 6 is invoked separately via RUN_SUMMARY below.
layered = run_all_layers(
    pdf,
    outdir="dbfs:/FileStore/employee_voice/output",
    run_topics=RUN_TOPICS,
    run_summary_layer=False,
)
pdf = layered["df"]
meta = layered["topic_meta"]

# COMMAND ----------

# Wrap back into Spark and write four outputs: fact, dim_topic, summary_category, summary_bu.
fact_schema = StructType([
    StructField("FeedbackID", StringType()),
    StructField("SurveyType", StringType()),
    StructField("SurveyDate", StringType()),  # keep as string to avoid pandas/Spark date tz friction
    StructField("BU", StringType()),
    StructField("Dept", StringType()),
    StructField("EmployeeGroup", StringType()),
    StructField("Comment", StringType()),
    StructField("CleanComment", StringType()),
    StructField("Sentiment", StringType()),
    StructField("SentimentScore", DoubleType()),
    StructField("Category1", StringType()),
    StructField("Category1Score", DoubleType()),
    StructField("Category2", StringType()),
    StructField("Category2Score", DoubleType()),
    StructField("AllCategories", StringType()),
    StructField("Emotion", StringType()),
    StructField("EmotionScore", DoubleType()),
    StructField("Topic", IntegerType()),
    StructField("TopicName", StringType()),
    StructField("TopicKeywords", StringType()),
    StructField("RiskScore", DoubleType()),
    StructField("RiskBand", StringType()),
])

# Cast columns that may have come back as object/NaN.
for col, dtype in (("SentimentScore", "float64"), ("Category1Score", "float64"),
                   ("Category2Score", "float64"), ("EmotionScore", "float64"),
                   ("RiskScore", "float64")):
    if col in pdf.columns:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce").fillna(0.0)
for col in ("Topic",):
    if col in pdf.columns:
        # Cast to Int64 first, then unwrap the pandas extension dtype into a
        # plain object column of native ints + None. PySpark's createDataFrame
        # does not understand pd.NA / pandas extension arrays in 4.x and would
        # otherwise coerce the column to float64 and fail the IntegerType
        # schema check. Converting here keeps the schema honest.
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce").astype("Int64")
        pdf[col] = pdf[col].astype("object").where(pdf[col].notna(), None)

# Cast SurveyDate to ISO string for Spark.
if "SurveyDate" in pdf.columns:
    pdf["SurveyDate"] = pd.to_datetime(pdf["SurveyDate"], errors="coerce").dt.strftime("%Y-%m-%d")

fact_pdf = pdf[[f.name for f in fact_schema.fields if f.name in pdf.columns]].copy()
# Add any missing columns with nulls so Spark schema matches.
for f in fact_schema.fields:
    if f.name not in fact_pdf.columns:
        fact_pdf[f.name] = None

fact_pdf = fact_pdf[[f.name for f in fact_schema.fields]]
fact_sdf = spark.createDataFrame(fact_pdf, schema=fact_schema)

# Write the fact table.
if OUTPUT_CATALOG:
    fact_table = f"{OUTPUT_CATALOG}.{OUTPUT_DATABASE}.fact_employee_feedback"
    (fact_sdf.write
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .format("delta").saveAsTable(fact_table))
    print(f"Wrote Delta table: {fact_table}")
else:
    fact_path = "dbfs:/FileStore/employee_voice/output/fact_employee_feedback"
    (fact_sdf.write.mode("overwrite").format("delta").save(fact_path))
    print(f"Wrote Delta at: {fact_path}")

# COMMAND ----------

# dim_topic.csv equivalent (small, written from driver).
dim_rows = [
    {"Topic": int(tid), "TopicName": m["Name"],
     "TopicKeywords": ",".join(m["Keywords"]), "DocCount": int(m["Count"])}
    for tid, m in sorted(meta.items())
]
dim_pdf = pd.DataFrame(dim_rows)
if dim_pdf.empty:
    dim_pdf = pd.DataFrame(columns=["Topic", "TopicName", "TopicKeywords", "DocCount"])

if OUTPUT_CATALOG:
    dim_table = f"{OUTPUT_CATALOG}.{OUTPUT_DATABASE}.dim_topic"
    (spark.createDataFrame(dim_pdf)
          .write.mode("overwrite").option("overwriteSchema", "true")
          .format("delta").saveAsTable(dim_table))
    print(f"Wrote Delta table: {dim_table}")
else:
    dim_path = "dbfs:/FileStore/employee_voice/output/dim_topic"
    (spark.createDataFrame(dim_pdf).coalesce(1)
          .write.mode("overwrite").format("csv")
          .option("header", "true").save(dim_path))
    print(f"Wrote CSV at: {dim_path}")

# COMMAND ----------

# Spark-native summaries — these aggregations *should* run on Spark to scale.
summary_cat_sdf = (
    fact_sdf
    .groupBy("Category1", "Sentiment")
    .agg(F.count("FeedbackID").alias("FeedbackCount"))
    .groupBy("Category1")
    .pivot("Sentiment")
    .sum("FeedbackCount")
    .na.fill(0)
)
total_cols = [c for c in summary_cat_sdf.columns if c != "Category1"]
if total_cols:
    summary_cat_sdf = summary_cat_sdf.withColumn(
        "Total", sum(F.col(c) for c in total_cols)
    ).orderBy(F.col("Total").desc())
summary_cat_sdf.show(truncate=False)

if OUTPUT_CATALOG:
    cat_table = f"{OUTPUT_CATALOG}.{OUTPUT_DATABASE}.summary_category"
    (summary_cat_sdf.write.mode("overwrite").option("overwriteSchema", "true")
     .format("delta").saveAsTable(cat_table))
    print(f"Wrote Delta table: {cat_table}")
else:
    cat_path = "dbfs:/FileStore/employee_voice/output/summary_category"
    (summary_cat_sdf.coalesce(1).write.mode("overwrite").format("csv")
     .option("header", "true").save(cat_path))
    print(f"Wrote CSV at: {cat_path}")

# COMMAND ----------

if "BU" in fact_sdf.columns:
    summary_bu_sdf = (
        fact_sdf.groupBy("BU").agg(
            F.count("FeedbackID").alias("FeedbackCount"),
            F.round(F.avg("RiskScore"), 2).alias("AvgRiskScore"),
            F.sum(F.when(F.col("RiskBand") == "High", 1).otherwise(0)).alias("HighRiskCount"),
            F.sum(F.when(F.col("Sentiment") == "negative", 1).otherwise(0)).alias("NegativeCount"),
        ).orderBy(F.col("AvgRiskScore").desc())
    )
    summary_bu_sdf.show(truncate=False)

    if OUTPUT_CATALOG:
        bu_table = f"{OUTPUT_CATALOG}.{OUTPUT_DATABASE}.summary_bu"
        (summary_bu_sdf.write.mode("overwrite").option("overwriteSchema", "true")
         .format("delta").saveAsTable(bu_table))
        print(f"Wrote Delta table: {bu_table}")
    else:
        bu_path = "dbfs:/FileStore/employee_voice/output/summary_bu"
        (summary_bu_sdf.coalesce(1).write.mode("overwrite").format("csv")
         .option("header", "true").save(bu_path))
        print(f"Wrote CSV at: {bu_path}")

# COMMAND ----------

# Optional: LLM executive summary — only if Azure OpenAI creds are in scope as
# Databricks secrets. Wire the secret scope once via:
#   databricks secrets create-scope --scope hr-analytics
#   databricks secrets put --scope hr-analytics --key azure-openai-endpoint
#   databricks secrets put --scope hr-analytics --key azure-openai-key
#   databricks secrets put --scope hr-analytics --key azure-openai-deployment
# Then set RUN_SUMMARY=true in the widget.
if RUN_SUMMARY:
    try:
        os.environ["AZURE_OPENAI_ENDPOINT"]    = dbutils.secrets.get("hr-analytics", "azure-openai-endpoint")
        os.environ["AZURE_OPENAI_API_KEY"]     = dbutils.secrets.get("hr-analytics", "azure-openai-key")
        os.environ["AZURE_OPENAI_DEPLOYMENT"]  = dbutils.secrets.get("hr-analytics", "azure-openai-deployment")
        from employee_voice.summarizer import generate_summary
        out_path = generate_summary(fact_pdf, "dbfs:/FileStore/employee_voice/output")
        print(f"Exec summary written to: {out_path}")
    except Exception as exc:
        print(f"Skipped exec summary: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What just ran
# MAGIC
# MAGIC - Read input (CSV / JSON / Parquet / XLSX / Delta)
# MAGIC - Normalised column names + auto-detected the comment column
# MAGIC - Cleaned + filtered N/A noise
# MAGIC - Ran sentiment / category / emotion / topic discovery
# MAGIC - Computed the 0–100 attrition risk score + band
# MAGIC - Wrote four artifacts: `fact_employee_feedback`, `dim_topic`, `summary_category`, `summary_bu`
# MAGIC   - Delta tables when `output_catalog` is set
# MAGIC   - CSV/Delta files under `dbfs:/FileStore/employee_voice/output/` otherwise
# MAGIC
# MAGIC ### Scaling past ~1M rows
# MAGIC
# MAGIC The driver-side analysis loop (`toPandas() → analyze_*`) is fine for tens
# MAGIC of thousands of rows but doesn't parallelise across executors. To scale,
# MAGIC refactor each `analyze_*` call into a Pandas UDF:
# MAGIC
# MAGIC ```python
# MAGIC @pandas_udf("sentiment string, score double")
# MAGIC def sentiment_udf(texts: pd.Series) -> pd.DataFrame:
# MAGIC     from employee_voice.analyzers import analyze_sentiment
# MAGIC     df = analyze_sentiment(texts.tolist())
# MAGIC     df.columns = ["sentiment", "score"]
# MAGIC     return df
# MAGIC
# MAGIC fact_sdf = fact_sdf.withColumn(
# MAGIC     "_sentiment", sentiment_udf("CleanComment")
# MAGIC ).select("*", "_sentiment.sentiment", "_sentiment.score").drop("_sentiment")
# MAGIC ```
# MAGIC
# MAGIC Each executor then loads its own HF pipeline (or, cheaper, the fallback
# MAGIC engines are already stateless so they're safe to broadcast as a
# MAGIC `sc.broadcast()`-able object).
