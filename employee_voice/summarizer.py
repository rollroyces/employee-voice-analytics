"""
Optional executive summary using Azure OpenAI (or any OpenAI-compatible endpoint).
Disabled unless the four AZURE_OPENAI_* env vars are set.

Generates one short markdown brief + 3 follow-up talking-point questions, written
to <outdir>/summary/exec_summary.md.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Optional

import pandas as pd

from . import config as cfg

log = logging.getLogger(__name__)


def _df_to_payload(df: pd.DataFrame, max_rows_per_group: int = 50) -> str:
    """
    Build a compact text payload from the fact table for the LLM.
    Avoids sending raw comments back — only categories, sentiments, and risk bands.
    """
    df = df.copy()
    df["Sentiment"] = df["Sentiment"].fillna("No Comment")

    # Risk distribution
    risk = df["RiskBand"].value_counts().to_dict()
    # Sentiment distribution
    sent = df["Sentiment"].value_counts().to_dict()
    # Top categories
    top_cats = df["Category1"].value_counts().head(10).to_dict()
    # BU x high-risk
    bu_risk = (
        df.groupby("BU")
          .agg(HighRisk=("RiskBand", lambda s: int((s == "High").sum())),
               Avg=("RiskScore", "mean"),
               N=("FeedbackID", "count"))
          .reset_index()
          .sort_values("Avg", ascending=False)
          .head(10)
          .to_dict(orient="records")
    )
    # Intent-to-leave count
    intent = int(df["Comment"].fillna("").str.contains(
        "|".join(cfg.INTENT_TO_LEAVE_KEYWORDS), case=False, regex=True
    ).sum())

    payload = {
        "total_feedback": int(len(df)),
        "sentiment_distribution": sent,
        "risk_band_distribution": risk,
        "top_categories": top_cats,
        "bu_risk_summary": bu_risk,
        "intent_to_leave_count": intent,
    }
    return json.dumps(payload, indent=2, default=str)


def generate_summary(fact: pd.DataFrame, outdir: str) -> Optional[str]:
    """Returns path to the written summary, or None if not configured."""
    az = cfg.AZURE_OPENAI
    if not (az["endpoint"] and az["api_key"] and az["deployment"]):
        log.info("Azure OpenAI not configured — skipping exec summary")
        return None

    try:
        from openai import AzureOpenAI
    except ImportError:
        log.warning("openai package not installed — skipping exec summary")
        return None

    payload = _df_to_payload(fact)
    prompt = (
        "You are an HR analytics assistant. Given the aggregated metrics below, "
        "produce:\n"
        "1) A 5-7 bullet executive summary of what the data shows.\n"
        "2) Three follow-up questions HR leaders should ask in the next staff meeting.\n"
        "Be specific. Cite numbers. No marketing fluff.\n\n"
        f"DATA:\n{payload}"
    )

    client = AzureOpenAI(
        azure_endpoint=az["endpoint"],
        api_key=az["api_key"],
        api_version=az["api_version"],
    )
    resp = client.chat.completions.create(
        model=az["deployment"],
        messages=[
            {"role": "system", "content": "You write concise, evidence-grounded HR briefings."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""

    out_dir = os.path.join(outdir, cfg.SUMMARY_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exec_summary.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# Employee Voice — Executive Summary\n\n")
        fh.write(content.strip() + "\n")
    return out_path
