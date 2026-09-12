"""
Preprocessing: text cleaning and N/A noise filtering.

A comment that is just punctuation, an em-dash, or "no comment" gets marked
"Filtered" and excluded from sentiment/category scoring, but is still kept
in the output table with sentiment = "No Comment".
"""
from __future__ import annotations
import re
import pandas as pd

from .config import NA_PATTERNS, MIN_COMMENT_LEN


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs / non-printables, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    s = _URL_RE.sub(" ", s)
    s = _NON_PRINTABLE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def is_non_answer(text: str) -> bool:
    """True if the cleaned text is essentially empty / a known N/A token."""
    if not text:
        return True
    norm = text.strip().lower().rstrip(".!?")
    if norm in NA_PATTERNS:
        return True
    # purely punctuation / digits-only
    if re.fullmatch(r"[\W_]+", norm):
        return True
    if norm.isdigit():
        return True
    return False


def preprocess_series(series: pd.Series) -> pd.DataFrame:
    """
    Returns a DataFrame with:
        CleanComment (str)
        IsNonAnswer  (bool)  — caller routes these to "No Comment"
        IsShort      (bool)  — below MIN_COMMENT_LEN
    """
    cleaned = series.fillna("").astype(str).map(clean_text)
    is_na = cleaned.map(is_non_answer).astype(bool)
    # Cast the length comparison to a plain numpy bool array so the
    # subsequent `&` works on empty Arrow-backed string Series in
    # pandas 3.0. (The error in pandas 3.0 with Arrow strings is
    # `TypeError: operation 'and_' not supported for dtype 'str' with
    # dtype 'bool'` even though both operands *look* boolean.)
    lengths = cleaned.str.len().to_numpy()
    is_short = (lengths < MIN_COMMENT_LEN) & (~is_na.to_numpy())
    return pd.DataFrame({
        "CleanComment": cleaned,
        "IsNonAnswer": is_na,
        "IsShort": pd.Series(is_short, index=cleaned.index, dtype=bool),
    })
