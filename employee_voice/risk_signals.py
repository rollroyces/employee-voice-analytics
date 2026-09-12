"""
Push / pull factors and risk velocity for the Employee Voice pipeline.

Push factors  — things that drive an employee AWAY from the company
                (push = dissatisfaction with current role).
Pull factors  — things that attract them OUTWARD (pull = active
                interest in leaving, external opportunities, etc.).

Risk velocity — q-over-q change in negative-sentiment rate per
                stable EmployeeID. Requires (a) an EmployeeID column
                and (b) a SurveyDate column. If either is missing
                the column is null.

Each factor is detected by a list of keywords/phrases. The output
columns are:
  PushFactors  - comma-separated list of matched push factors ("" if none)
  PullFactors  - comma-separated list of matched pull factors
  RiskVelocity - float, change in negative-sentiment rate vs prior
                 quarter for the same employee. Null if computable.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------
PUSH_FACTORS: Dict[str, List[str]] = {
    "burnout": [
        "burned out", "burnt out", "burnout", "exhausted", "exhaustion",
        "overworked", "overwork", "no work-life", "no work life",
        "killing my work-life", "killing my work life",
        "late nights", "weekend work", "weekend emails", "always on",
    ],
    "workload": [
        "too much work", "too much on my plate", "understaffed", "no headcount",
        "under resourced", "under-resourced", "stretched thin", "too many deadlines",
    ],
    "manager": [
        "my manager", "my boss", "my supervisor", "toxic manager",
        "micromanaged", "micromanager", "no support from manager",
        "manager is unresponsive", "manager doesn't listen", "bad manager",
    ],
    "compensation": [
        "underpaid", "below market", "below market pay", "not paid enough",
        "no raise", "no bonus", "compensation is low", "salary is low",
    ],
    "career": [
        "no growth", "no promotion", "stuck", "dead end", "dead-end",
        "no path forward", "no career path", "career stagnation",
        "no learning", "no development", "no training",
    ],
    "culture": [
        "toxic culture", "toxic environment", "no values", "no transparency",
        "backstabbing", "political", "broken culture",
    ],
    "leadership": [
        "no vision", "vision keeps changing", "strategy is unclear",
        "leadership is lost", "executives don't care", "constant reorg",
        "layoffs", "restructuring",
    ],
}

PULL_FACTORS: Dict[str, List[str]] = {
    "intent_to_leave": [
        "i quit", "i resigned", "i am leaving", "i'm leaving", "resigning",
        "last day", "on my way out", "done with this place", "had enough",
        "fed up",
    ],
    "external_offer": [
        "accepted an offer", "got an offer", "received an offer",
        "offer from another company", "competing offer",
    ],
    "job_search": [
        "looking for a new job", "looking for another job", "looking for new job",
        "looking for another role", "interviewing",
        "interviews lined up", "talking to recruiters", "in final rounds",
    ],
    "competing_pull": [
        "better pay elsewhere", "more interesting role elsewhere",
        "company down the street", "startup", "competitor offered",
    ],
    "personal": [
        "moving away", "relocating", "going back to school", "career break",
        "sabbatical", "family reasons",
    ],
}


def detect_factors(text: str, factors: Dict[str, List[str]]) -> List[str]:
    """Return the factor names whose keywords appear in `text`."""
    if not text:
        return []
    text_l = text.lower()
    hits: List[str] = []
    for factor, keywords in factors.items():
        if any(kw in text_l for kw in keywords):
            hits.append(factor)
    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def add_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add PushFactors and PullFactors columns to a fact DataFrame.
    Uses the CleanComment column if present, else Comment.

    Factor lists are comma-separated strings for portability into
    Parquet/Delta; the in-pipeline form is a Python list, but we
    stringify to keep downstream BI tools happy.
    """
    out = df.copy()
    text_col = "CleanComment" if "CleanComment" in out.columns else "Comment"
    if text_col not in out.columns:
        log.warning("No Comment / CleanComment column; factor detection skipped")
        out["PushFactors"] = ""
        out["PullFactors"] = ""
        return out

    def _push(t: str) -> str:
        return ",".join(detect_factors(t, PUSH_FACTORS))

    def _pull(t: str) -> str:
        return ",".join(detect_factors(t, PULL_FACTORS))

    out["PushFactors"] = out[text_col].fillna("").astype(str).map(_push)
    out["PullFactors"] = out[text_col].fillna("").astype(str).map(_pull)
    return out


def compute_risk_velocity(
    df: pd.DataFrame,
    *,
    employee_col: str = "EmployeeID",
    date_col: str = "SurveyDate",
    sentiment_col: str = "Sentiment",
    negative_label: str = "negative",
) -> pd.DataFrame:
    """Compute q-over-q change in negative-sentiment rate per employee.

    Returns a copy of `df` with an extra `RiskVelocity` column. The
    column is the difference between the current row's negative-sent
    indicator and the prior quarter's, where "quarter" is the calendar
    quarter of SurveyDate.

    Requirements:
      - `df` must have a stable EmployeeID column and a parseable
        SurveyDate column. If either is missing, the column is all null
        and a warning is logged.
      - Multiple rows in the same quarter for the same employee are
        averaged.

    The velocity is bounded in [-1, 1] so a single comment can move the
    needle at most from fully positive (0 negatives) to fully negative
    (all negatives).
    """
    out = df.copy()
    if employee_col not in out.columns or date_col not in out.columns:
        log.warning(
            "compute_risk_velocity needs columns %r and %r; column missing",
            employee_col, date_col,
        )
        out["RiskVelocity"] = np.nan
        return out

    if sentiment_col not in out.columns:
        out["RiskVelocity"] = np.nan
        return out

    # Parse SurveyDate to datetime. Rows that fail to parse get NaT.
    parsed = pd.to_datetime(out[date_col], errors="coerce")
    if parsed.isna().all():
        log.warning("SurveyDate column has no parseable values; RiskVelocity = null")
        out["RiskVelocity"] = np.nan
        return out

    # Quarter as a period string e.g. "2026Q1" for stable sorting.
    quarter = parsed.dt.to_period("Q").astype("string")
    neg = (out[sentiment_col] == negative_label).astype(float)

    work = pd.DataFrame({
        "EmployeeID": out[employee_col].fillna("__NA__").astype(str),
        "Quarter": quarter,
        "NegativeInd": neg,
    })
    # Aggregate: mean negative-sent rate per (employee, quarter).
    agg = work.groupby(["EmployeeID", "Quarter"], dropna=False)["NegativeInd"].mean()
    # Prior quarter per employee.
    agg = agg.reset_index()
    agg["PriorQuarterRate"] = agg.groupby("EmployeeID")["NegativeInd"].shift(1)
    agg["Velocity"] = (agg["NegativeInd"] - agg["PriorQuarterRate"]).clip(-1.0, 1.0)

    # Map back onto out via (EmployeeID, Quarter) join.
    out["__Quarter__"] = quarter
    out["__EmployeeID__"] = work["EmployeeID"].values
    lookup = agg.set_index(["EmployeeID", "Quarter"])["Velocity"]
    out["RiskVelocity"] = out.set_index(["__EmployeeID__", "__Quarter__"]).index.map(lookup).values

    out = out.drop(columns=["__Quarter__", "__EmployeeID__"])
    return out
