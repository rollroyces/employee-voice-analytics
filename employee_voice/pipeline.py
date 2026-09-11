"""
Orchestration: load input -> preprocess -> analyze -> score -> emit fact + dims.
"""
from __future__ import annotations
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from . import config as cfg
from .preprocess import preprocess_series
from .analyzers import (
    analyze_sentiment,
    analyze_category,
    analyze_emotion,
    score_risk,
)
from .topics import discover_topics

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
        run_topics: bool = True) -> dict:
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

    # preprocess
    pp = preprocess_series(out["Comment"])
    out["CleanComment"] = pp["CleanComment"]
    is_non_answer = pp["IsNonAnswer"]
    is_short = pp["IsShort"]
    analyzable_mask = (~is_non_answer) & (~is_short)
    log.info("Analyzable rows: %d / %d", int(analyzable_mask.sum()), len(out))

    # sentiment / category / emotion over analyzable rows
    if analyzable_mask.any():
        idx = out.index[analyzable_mask]
        texts = out.loc[idx, "CleanComment"].tolist()

        sent = analyze_sentiment(texts)
        sent.index = idx
        cat = analyze_category(texts)
        cat.index = idx
        emo = analyze_emotion(texts)
        emo.index = idx

        out = pd.concat([out, sent, cat, emo], axis=1)
    else:
        out["Sentiment"] = np.nan
        out["SentimentScore"] = np.nan
        out["Category1"] = np.nan
        out["Category1Score"] = np.nan
        out["Category2"] = np.nan
        out["Category2Score"] = np.nan
        out["AllCategories"] = np.nan
        out["Emotion"] = np.nan
        out["EmotionScore"] = np.nan

    # fill non-analyzable
    for col, default in (
        ("Sentiment", "No Comment"),
        ("SentimentScore", 0.0),
        ("Category1", "No Comment"),
        ("Category1Score", 0.0),
        ("Category2", ""),
        ("Category2Score", 0.0),
        ("AllCategories", ""),
        ("Emotion", "No Comment"),
        ("EmotionScore", 0.0),
    ):
        out[col] = out[col].astype(object).where(analyzable_mask, default)

    # risk score (computed for every row; non-answers just stay Low)
    risks = []
    for _, row in out.iterrows():
        if not analyzable_mask.loc[_]:
            risks.append((0.0, "Low"))
            continue
        score, band = score_risk(
            sentiment=str(row["Sentiment"]),
            sentiment_score=float(row["SentimentScore"] or 0),
            emotion=str(row["Emotion"] or ""),
            emotion_score=float(row["EmotionScore"] or 0),
            all_categories=str(row["AllCategories"] or ""),
            comment=str(row["CleanComment"] or ""),
        )
        risks.append((score, band))
    out["RiskScore"] = [r[0] for r in risks]
    out["RiskBand"] = [r[1] for r in risks]

    # topics (only on analyzable comments)
    if run_topics:
        if analyzable_mask.any():
            tdf, meta = discover_topics(out.loc[analyzable_mask, "CleanComment"].tolist())
            tdf.index = out.index[analyzable_mask]
            out["Topic"] = np.nan
            out["TopicName"] = ""
            out["TopicKeywords"] = ""
            for col in ("Topic", "TopicName", "TopicKeywords"):
                out.loc[tdf.index, col] = tdf[col].values
        else:
            out["Topic"] = np.nan
            out["TopicName"] = ""
            out["TopicKeywords"] = ""
            meta = {}
    else:
        out["Topic"] = np.nan
        out["TopicName"] = ""
        out["TopicKeywords"] = ""
        meta = {}

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
    pd.DataFrame(rows).to_csv(dim_topic_path, index=False)

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
