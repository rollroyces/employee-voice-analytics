"""
Orchestration: load input -> run all layers -> emit fact + dims.

Layer execution is delegated to `employee_voice.layers.run_all_layers`
which enforces the per-layer schema contract in LAYER_CONTRACTS (config.py).
This module is responsible only for:
  - input parsing / column normalisation
  - calling the layer runner
  - writing the four output artifacts
  - the optional LLM exec summary
"""
from __future__ import annotations
import logging
import os
from typing import Optional

import pandas as pd

from . import config as cfg
from .layers import run_all_layers

log = logging.getLogger(__name__)

REQUIRED_INPUT_COLS_HINT = (
    "feedbackid", "surveytype", "surveydate", "bu", "dept", "employeegroup",
    "comment",
)


def _read_input(path: str, sheet: Optional[str] = None,
                text_column: Optional[str] = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet or 0)
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    log.info("Loaded %d rows from %s", len(df), path)
    return df


def _normalize_columns(df: pd.DataFrame, text_column: Optional[str]) -> pd.DataFrame:
    """Lowercase headers, alias common variants, and rename the text column."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    alias_map = {}
    for col in df.columns:
        low = col.lower().replace(" ", "").replace("_", "")
        alias_map[col] = low
    df = df.rename(columns=alias_map)

    # common synonyms
    rename = {}
    for col in list(df.columns):
        if col in ("feedback_id", "id", "responseid"):
            rename[col] = "feedbackid"
        elif col in ("survey_type", "surveyname"):
            rename[col] = "surveytype"
        elif col in ("survey_date", "date", "completed_at", "completedat"):
            rename[col] = "surveydate"
        elif col in ("bu", "businessunit", "business"):
            rename[col] = "bu"
        elif col in ("department", "team"):
            rename[col] = "dept"
        elif col in ("employee_group", "employeegroup", "group", "tenuregroup"):
            rename[col] = "employeegroup"
    if rename:
        df = df.rename(columns=rename)

    # pick the comment column
    if text_column:
        if text_column not in df.columns:
            # try case-insensitive match
            for c in df.columns:
                if c.lower() == text_column.lower():
                    text_column = c
                    break
            else:
                raise ValueError(
                    f"--text-column '{text_column}' not found. "
                    f"Available: {list(df.columns)}"
                )
    else:
        for cand in ("comment", "comments", "feedback", "response",
                     "verbatim", "text", "answer"):
            if cand in df.columns:
                text_column = cand
                break
        if text_column is None:
            raise ValueError(
                "No comment column found. Pass --text-column explicitly. "
                f"Available: {list(df.columns)}"
            )

    df = df.rename(columns={text_column: "Comment"})
    return df


def _ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "FeedbackID" not in df.columns and "feedbackid" not in df.columns:
        df = df.copy()
        df.insert(0, "FeedbackID", [f"F{idx+1:06d}" for idx in range(len(df))])
    return df


def run(input_path: str,
        outdir: str,
        sheet: Optional[str] = None,
        text_column: Optional[str] = None,
        run_topics: bool = True,
        *,
        sentiment_backend: str = "auto",
        category_backend: str = "auto",
        emotion_backend: str = "auto",
        llm_options: Optional[dict] = None) -> dict:
    """End-to-end. Returns the artifacts dict so callers can inspect results."""
    os.makedirs(outdir, exist_ok=True)
    raw = _read_input(input_path, sheet=sheet, text_column=text_column)
    df = _normalize_columns(raw, text_column=text_column)
    df = _ensure_ids(df)

    # pass-through columns
    passthrough = [c for c in
                   ("FeedbackID", "feedbackid",
                    "SurveyType", "surveytype",
                    "SurveyDate", "surveydate",
                    "BU", "bu",
                    "Dept", "dept",
                    "EmployeeGroup", "employeegroup")
                   if c in df.columns]
    out = df[passthrough].copy()

    # canonicalise column casing
    rename_back = {}
    for c in out.columns:
        if c.lower() in ("feedbackid",):
            rename_back[c] = "FeedbackID"
        elif c.lower() == "surveytype":
            rename_back[c] = "SurveyType"
        elif c.lower() == "surveydate":
            rename_back[c] = "SurveyDate"
        elif c.lower() == "bu":
            rename_back[c] = "BU"
        elif c.lower() == "dept":
            rename_back[c] = "Dept"
        elif c.lower() == "employeegroup":
            rename_back[c] = "EmployeeGroup"
    if rename_back:
        out = out.rename(columns=rename_back)

    out["Comment"] = df["Comment"].fillna("").astype(str)

    # Run Layers 0-5 (and optionally 4 / 6) under the contract in
    # config.LAYER_CONTRACTS. Raises LayerDependencyError on missing deps
    # and LayerContractError on schema violations.
    layered = run_all_layers(
        out,
        outdir=outdir,
        run_topics=run_topics,
        run_summary_layer=False,  # CLI handles Layer 6 separately
        sentiment_backend=sentiment_backend,
        category_backend=category_backend,
        emotion_backend=emotion_backend,
        llm_options=llm_options,
    )
    out = layered["df"]
    meta = layered["topic_meta"]

    # write outputs
    fact_path = os.path.join(outdir, cfg.OUTPUT_FACT)
    out.to_csv(fact_path, index=False)

    dim_topic_path = os.path.join(outdir, cfg.OUTPUT_DIM_TOPIC)
    rows = []
    for tid, m in sorted(meta.items()):
        rows.append({
            "Topic": tid,
            "TopicName": m["Name"],
            "TopicKeywords": ",".join(m["Keywords"]),
            "DocCount": m["Count"],
        })
    # Explicit columns so an empty `rows` still produces a header-only CSV
    # that can be round-tripped through pd.read_csv without EmptyDataError.
    pd.DataFrame(rows, columns=["Topic", "TopicName", "TopicKeywords", "DocCount"])\
        .to_csv(dim_topic_path, index=False)

    summary_path = os.path.join(outdir, cfg.OUTPUT_SUMMARY_CATEGORY)
    summary = (
        out.assign(Sentiment=out["Sentiment"].fillna("No Comment"))
        .pivot_table(index="Category1", columns="Sentiment",
                     values="FeedbackID", aggfunc="count", fill_value=0)
        .reset_index()
    )
    summary["Total"] = summary.iloc[:, 1:].sum(axis=1)
    summary = summary.sort_values("Total", ascending=False)
    summary.to_csv(summary_path, index=False)

    bu_path = os.path.join(outdir, cfg.OUTPUT_SUMMARY_BU)
    if "BU" in out.columns:
        bu_summary = (
            out.groupby("BU")
            .agg(
                FeedbackCount=("FeedbackID", "count"),
                AvgRiskScore=("RiskScore", "mean"),
                HighRiskCount=("RiskBand", lambda s: int((s == "High").sum())),
                NegativeCount=("Sentiment", lambda s: int((s == "negative").sum())),
            )
            .reset_index()
            .sort_values("AvgRiskScore", ascending=False)
        )
        bu_summary.to_csv(bu_path, index=False)

    return {
        "fact_path": fact_path,
        "dim_topic_path": dim_topic_path,
        "summary_path": summary_path,
        "bu_path": bu_path if "BU" in out.columns else None,
        "fact": out,
        "topic_meta": meta,
    }
