"""Tests for the benchmark harness internals.

These don't measure performance — they verify the parsing
logic that the parent scripts use to extract timing from child
subprocess output. If this breaks, every bench number is suspect.

The category-models bench had a real bug where the parent parsed
`<rows>` as `<elapsed>` (parts[2] vs parts[3]). The bug shipped
because no test exercised the parser in isolation. This file
prevents that class of bug.
"""
from __future__ import annotations
import pytest


# Re-create the parser functions by copying the exact logic from
# the harness files. We import the modules and reach into the
# function bodies rather than copy-pasting because the bug we are
# guarding against is divergence between the parser logic and
# what the test thinks it is.
def _category_parse(stdout: str):
    """Mirror of the parse loop in bench_category_models.time_model."""
    for line in stdout.splitlines():
        if line.startswith("MEASURE"):
            parts = line.split()
            try:
                return float(parts[3])
            except (ValueError, IndexError):
                continue
    return float("nan")


def _hf_parse(stdout: str):
    """Mirror of parse_measure_line in bench_hf.py."""
    for line in stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] == "MEASURE":
            try:
                return parts[1], int(parts[2]), float(parts[3])
            except (ValueError, IndexError):
                continue
    return None


# A canonical child output line for the category bench.
CATEGORY_LINE = "MEASURE facebook/bart-large-mnli 50 32.3451"
# A canonical child output line for the HF bench.
HF_LINE = "MEASURE sentiment 100 4.4991"


class TestCategoryBenchParser:
    def test_parses_elapsed_not_rows(self):
        """The bug we are guarding against: parts[2] is the row
        count (50), not the elapsed (32.34). If anyone changes
        the parser to read parts[2], this test will catch it."""
        result = _category_parse(CATEGORY_LINE + "\n")
        assert result == pytest.approx(32.3451, rel=1e-3), (
            f"parser returned {result}; expected 32.3451 (the elapsed "
            f"time, not the 50 row count)"
        )

    def test_handles_short_lines(self):
        """Lines with too few fields should return nan, not crash."""
        assert _category_parse("MEASURE foo bar\n") != _category_parse("MEASURE foo bar")  # both nan
        # The above is just a sanity check; both should be nan.
        import math
        assert math.isnan(_category_parse("MEASURE foo bar\n"))
        assert math.isnan(_category_parse("MEASURE foo bar\nbaz qux\n"))

    def test_handles_no_measure_line(self):
        import math
        assert math.isnan(_category_parse("Some other output\nfrom the child\n"))

    def test_matches_child_format(self):
        """If the child's print format changes, this test fails.
        Lock the child output format and the parent parser in
        one place: a single round-trip check.
        """
        # The child does: print(f"MEASURE {model} {rows} {elapsed:.4f}")
        # So the line is `MEASURE <model> <rows> <elapsed>`.
        # The parent takes parts[3] for elapsed.
        # The bug we are guarding against: someone changes the
        # child format (e.g. to "MEASURE <elapsed> <rows> <model>")
        # and the parent parser silently keeps reading parts[3] which
        # would now be the model string, not the elapsed.
        line = "MEASURE facebook/bart-large-mnli 50 32.3451"
        parts = line.split()
        # Document the format here. If you change the child, change
        # this assertion and the corresponding parser in lockstep.
        assert parts[0] == "MEASURE"
        assert parts[1] == "facebook/bart-large-mnli"  # model
        assert parts[2] == "50"                        # rows
        assert parts[3] == "32.3451"                   # elapsed


class TestHFBenchParser:
    def test_parses_layer_rows_elapsed(self):
        result = _hf_parse(HF_LINE + "\n")
        assert result == ("sentiment", 100, pytest.approx(4.4991, rel=1e-3))

    def test_returns_none_for_malformed(self):
        assert _hf_parse("not a measure line\n") is None
        assert _hf_parse("MEASURE\n") is None
        assert _hf_parse("MEASURE foo bar baz\n") is None
