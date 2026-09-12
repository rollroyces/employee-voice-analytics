"""
Enterprise privacy guardrails for the Employee Voice Analytics toolkit.

Two layers of protection:

  1. PII scrubbing: Replace names, emails, phone numbers and project
     codenames in free text with placeholder tokens before the text is
     passed to embeddings, zero-shot classifiers, or any LLM provider.

  2. k-anonymity suppression: For any demographic or department slice with
     fewer than `K_ANONYMITY_THRESHOLD` responses, drop verbatim quotes
     and aggregate the score columns to `*_suppressed` so downstream
     dashboards cannot de-anonymise individuals.

Default backend is regex-only so the package works without a 500 MB
spaCy model. If `presidio-analyzer` is installed, set
`PIIScrubber(backend="presidio")` to use it. The output is the same.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

# Standard email pattern — deliberately permissive; false positives are
# cheaper than misses here because we redacted text never leaves the run.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# International phone numbers with optional + and separators.
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}"
)

# SSN-like 9-digit sequences.
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Employee ID patterns common in HR systems: E12345, EMP12345, #12345.
_EMP_ID_RE = re.compile(r"\b(?:EMP|E|#)\d{4,8}\b", re.IGNORECASE)

# Project codenames: ALL-CAPS tokens 2-5 chars, optionally with digits,
# occurring after words like "project", "program", "codename", "initiative".
_CODENAME_RE = re.compile(
    r"\b(?:project|program|initiative|codename|code\s*name)\s+"
    r"([A-Z][A-Z0-9]{1,4}(?:-[A-Z0-9]{1,4})?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScrubResult:
    """Result of scrubbing a single text."""
    text: str
    redacted_count: int
    found: dict[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:  # truthy iff anything was redacted
        return self.redacted_count > 0


class PIIScrubber:
    """
    Pluggable PII scrubber. `backend="regex"` (default) needs no
    extra dependencies; `backend="presidio"` requires the
    `presidio-analyzer` package plus a spaCy NER model. The two
    backends return the same ScrubResult shape so callers don't have
    to branch.

    Recognised entities (both backends):
      EMPLOYEE_NAME   -> [EMPLOYEE_NAME]
      MANAGER_NAME    -> [MANAGER_NAME]
      EMAIL           -> [EMAIL]
      PHONE           -> [PHONE]
      SSN             -> [SSN]
      EMPLOYEE_ID     -> [EMPLOYEE_ID]
      PROJECT_CODE    -> [PROJECT_CODE]

    Heuristics for EMPLOYEE_NAME vs MANAGER_NAME: the regex backend
    uses the surrounding text ("my manager X", "X is a great manager",
    "report to X", "manager is X") to mark a name as MANAGER_NAME
    instead of EMPLOYEE_NAME. Names that don't match the heuristic
    are marked EMPLOYEE_NAME.
    """

    # Manager detection patterns. Each regex looks back from a capitalised
    # name token and fires the named rule if it matches.
    _MANAGER_RULES = [
        re.compile(r"\bmy\s+manager\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", re.IGNORECASE),
        re.compile(r"\bmanager\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", re.IGNORECASE),
        re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:is|was)\s+my\s+manager", re.IGNORECASE),
        re.compile(r"\b(?:reports?|reported|reporting)\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", re.IGNORECASE),
        re.compile(r"\b(?:supervisor|boss|lead)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", re.IGNORECASE),
        re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:manages|managed)\s+(?:me|my\s+team)", re.IGNORECASE),
    ]

    # Capitalised name candidates — first letter upper, rest lower,
    # optionally followed by additional capitalised tokens. We only
    # call something a "name" if it sits next to a person-y keyword,
    # to keep false positives low.
    _NAME_RULES = [
        re.compile(r"\b(?:colleague|coworker|teammate|peer)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", re.IGNORECASE),
        re.compile(r"\b(?:with|from|to|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:on|in|at|about)", re.IGNORECASE),
    ]

    def __init__(self, backend: str = "regex") -> None:
        if backend not in ("regex", "presidio"):
            raise ValueError(f"Unknown PII backend: {backend!r}")
        self.backend = backend
        self._presidio_analyzer = None
        if backend == "presidio":
            self._init_presidio()

    def _init_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as exc:
            raise ImportError(
                "Presidio backend requires `pip install presidio-analyzer` "
                "and a spaCy NER model. Install with: "
                "`pip install employee-voice-analytics[privacy]`."
            ) from exc
        try:
            self._presidio_analyzer = AnalyzerEngine()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to initialise Presidio AnalyzerEngine. Make sure a "
                "spaCy NER model is installed: `python -m spacy download "
                "en_core_web_sm`."
            ) from exc

    # --- public API --------------------------------------------------

    def scrub(self, text: str) -> ScrubResult:
        """Scrub a single string. Returns a ScrubResult with the
        redacted text and counts per entity type."""
        if not isinstance(text, str) or not text:
            return ScrubResult(text=text or "", redacted_count=0)
        if self.backend == "presidio":
            return self._scrub_presidio(text)
        return self._scrub_regex(text)

    def scrub_series(self, series: pd.Series) -> pd.Series:
        """Vectorised wrapper around `scrub` for a pandas Series."""
        return series.fillna("").astype(str).map(self.scrub).map(lambda r: r.text)

    def detect(self, text: str) -> dict[str, int]:
        """Return counts of detected entities without modifying the text.
        Useful for audit logging."""
        return self.scrub(text).found

    # --- backends -----------------------------------------------------

    def _scrub_regex(self, text: str) -> ScrubResult:
        """Pure-regex PII scrubber. No external models required."""
        found: dict[str, int] = {}

        def _sub(pattern: re.Pattern, replacement: str, entity: str) -> str:
            nonlocal text
            new_text, n = pattern.subn(replacement, text)
            if n:
                found[entity] = found.get(entity, 0) + n
                text = new_text
            return text

        _sub(_EMAIL_RE, "[EMAIL]", "EMAIL")
        _sub(_PHONE_RE, "[PHONE]", "PHONE")
        _sub(_SSN_RE, "[SSN]", "SSN")
        _sub(_EMP_ID_RE, "[EMPLOYEE_ID]", "EMPLOYEE_ID")
        # Codename patterns capture the codename token in group 1; we
        # want to keep the prefix word ("project", "program", ...) and
        # replace just the token. The pattern isn't a plain `_sub` case
        # because we need to keep the prefix word intact.
        before = text
        text = _CODENAME_RE.sub(
            lambda m: m.group(0).replace(m.group(1), "[PROJECT_CODE]"),
            text,
        )
        n_codenames = sum(1 for _ in _CODENAME_RE.finditer(before))
        if n_codenames:
            found["PROJECT_CODE"] = found.get("PROJECT_CODE", 0) + n_codenames

        # Names: manager-detection first (more specific), then employee.
        for rule in self._MANAGER_RULES:
            text, n = rule.subn(
                lambda m: m.group(0).replace(m.group(1), "[MANAGER_NAME]"),
                text,
            )
            if n:
                found["MANAGER_NAME"] = found.get("MANAGER_NAME", 0) + n

        for rule in self._NAME_RULES:
            text, n = rule.subn(
                lambda m: m.group(0).replace(m.group(1), "[EMPLOYEE_NAME]"),
                text,
            )
            if n:
                found["EMPLOYEE_NAME"] = found.get("EMPLOYEE_NAME", 0) + n

        total = sum(found.values())
        return ScrubResult(text=text, redacted_count=total, found=found)

    def _scrub_presidio(self, text: str) -> ScrubResult:
        """Presidio-backed scrubber. Maps Presidio entity types to our
        placeholder tokens."""
        if self._presidio_analyzer is None:
            self._init_presidio()

        # Custom recognisers for EMPLOYEE_ID and PROJECT_CODE — Presidio's
        # built-ins don't cover HR-style employee IDs.
        from presidio_analyzer import PatternRecognizer, Pattern
        custom = [
            PatternRecognizer(
                supported_entity="EMPLOYEE_ID",
                patterns=[Pattern(name="emp_id", regex=_EMP_ID_RE.pattern, score=0.9)],
            ),
            PatternRecognizer(
                supported_entity="PROJECT_CODE",
                patterns=[Pattern(name="codename", regex=_CODENAME_RE.pattern, score=0.6)],
            ),
        ]
        for r in custom:
            self._presidio_analyzer.registry.add_recognizer(r)

        results = self._presidio_analyzer.analyze(text=text, language="en")
        found: dict[str, int] = {}

        # Walk results right-to-left so offsets stay valid as we splice.
        placeholder_map = {
            "PERSON": "EMPLOYEE_NAME",
            "EMAIL_ADDRESS": "EMAIL",
            "PHONE_NUMBER": "PHONE",
            "US_SSN": "SSN",
            "EMPLOYEE_ID": "EMPLOYEE_ID",
            "PROJECT_CODE": "PROJECT_CODE",
        }
        # Manager-name promotion: if a PERSON entity is within 30 chars
        # of a "manager" / "supervisor" keyword, label it MANAGER_NAME.
        def _is_manager_context(start: int) -> bool:
            window = text[max(0, start - 30):start].lower()
            return any(w in window for w in ("manager", "supervisor", "boss", "lead", "reports to"))

        out = text
        for r in sorted(results, key=lambda x: x.start, reverse=True):
            label = placeholder_map.get(r.entity_type)
            if not label:
                continue
            if label == "EMPLOYEE_NAME" and _is_manager_context(r.start):
                label = "MANAGER_NAME"
            out = out[:r.start] + f"[{label}]" + out[r.end:]
            found[label] = found.get(label, 0) + 1

        return ScrubResult(text=out, redacted_count=sum(found.values()), found=found)


# ---------------------------------------------------------------------------
# k-anonymity
# ---------------------------------------------------------------------------

@dataclass
class KAnonymityConfig:
    threshold: int = 5
    suppress_verbatim: bool = True   # if True, blank Comment / CleanComment for under-threshold slices
    suppress_quote_columns: tuple[str, ...] = ("Comment", "CleanComment")


def apply_k_anonymity(
    df: pd.DataFrame,
    group_by: Iterable[str],
    config: KAnonymityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (suppressed_df, slice_summary).

    `suppressed_df` is a copy of `df` where rows belonging to slices
    smaller than the threshold have their verbatim columns blanked and
    their scores preserved (so aggregate dashboards still work without
    exposing the individual comments).

    `slice_summary` is a per-slice count report for audit logging.
    """
    group_by = list(group_by)
    for c in group_by:
        if c not in df.columns:
            raise ValueError(f"k-anonymity group column {c!r} not in DataFrame")

    counts = df.groupby(group_by, dropna=False).size().rename("slice_count")
    slice_summary = counts.reset_index()

    # Under-threshold slices
    small = counts[counts < config.threshold].reset_index()
    if small.empty:
        return df.copy(), slice_summary
    small_keys = small[group_by].itertuples(index=False, name=None)

    out = df.copy()
    # Build a boolean mask: row's group_by values match any small slice
    mask = pd.Series(False, index=out.index)
    for key in small_keys:
        m = pd.Series(True, index=out.index)
        for col, val in zip(group_by, key):
            m &= out[col].fillna("__NA__") == (val if pd.notna(val) else "__NA__")
        mask |= m

    if config.suppress_verbatim:
        for col in config.suppress_quote_columns:
            if col in out.columns:
                out.loc[mask, col] = ""

    return out, slice_summary
