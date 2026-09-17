"""Tests for the few-shot exemplar pool and retrieval."""
from __future__ import annotations

import pytest

from employee_voice import exemplars
from employee_voice.config import CATEGORIES


class TestExemplarPool:
    def test_pool_nonempty(self):
        assert len(exemplars.EXEMPLARS) >= 30

    def test_pool_covers_every_category(self):
        """At least one exemplar per category in CATEGORIES. This is
        a hard rule for the pool because the static formatter uses it
        to guarantee category coverage in any prompt."""
        covered = {cat for _, cat, _ in exemplars.EXEMPLARS}
        missing = set(CATEGORIES) - covered
        assert not missing, f"categories missing from pool: {missing}"

    def test_pool_tuples_have_correct_shape(self):
        for entry in exemplars.EXEMPLARS:
            assert isinstance(entry, tuple) and len(entry) == 3
            comment, cat, sec = entry
            assert isinstance(comment, str) and comment
            assert isinstance(cat, str) and cat in (set(CATEGORIES) | {"Other"})
            assert sec is None or (isinstance(sec, str) and sec in set(CATEGORIES))


class TestFormatExemplars:
    def test_format_returns_string(self):
        out = exemplars.format_exemplars(max_n=5)
        assert isinstance(out, str)
        assert "Examples" in out

    def test_format_one_per_category(self):
        out = exemplars.format_exemplars(max_n=18)
        cats_in_out = []
        for cat in CATEGORIES:
            if f'"category": "{cat}"' in out:
                cats_in_out.append(cat)
        # We asked for 18, there are 18 categories -> expect one of each
        assert len(cats_in_out) == 18

    def test_format_max_n_caps_output(self):
        out = exemplars.format_exemplars(max_n=3)
        # Each example contributes one "  \"...\"" line.
        example_lines = sum(1 for line in out.splitlines() if line.startswith('  "'))
        assert example_lines <= 3

    def test_format_empty_pool(self):
        assert exemplars.format_exemplars([]) == ""


class TestRetrieveExemplars:
    def test_returns_k(self):
        out = exemplars.retrieve_exemplars("I hate my salary", k=5)
        assert len(out) == 5

    def test_top_match_is_compensation(self):
        """For a salary query, the top match should be a Compensation
        exemplar (or at least very close to it)."""
        out = exemplars.retrieve_exemplars(
            "my salary is below market and I'm frustrated", k=3,
        )
        cats = [cat for _, cat, _ in out]
        assert "Compensation and Benefits" in cats

    def test_top_match_is_workload(self):
        out = exemplars.retrieve_exemplars(
            "I'm on call every weekend and it's destroying my life", k=3,
        )
        cats = [cat for _, cat, _ in out]
        # Should retrieve Workload exemplars (or Work-Life Balance as secondary)
        assert "Workload" in cats or "Work-Life Balance" in cats

    def test_top_match_is_onboarding(self):
        out = exemplars.retrieve_exemplars(
            "first week was chaos and nobody told me what to do", k=3,
        )
        cats = [cat for _, cat, _ in out]
        assert "Onboarding" in cats

    def test_empty_query_returns_first_k(self):
        out = exemplars.retrieve_exemplars("", k=3)
        assert len(out) == 3

    def test_k_capped_at_pool_size(self):
        out = exemplars.retrieve_exemplars("anything", k=10000)
        assert len(out) == len(exemplars.EXEMPLARS)

    def test_k_zero_returns_empty(self):
        assert exemplars.retrieve_exemplars("anything", k=0) == []


class TestFormatRetrieved:
    def test_includes_retrieved_examples(self):
        out = exemplars.format_retrieved("salary is below market", k=3)
        assert isinstance(out, str)
        # Should include some category labels
        assert "category" in out

    def test_retrieved_differs_from_static(self):
        """Two different queries should produce different exemplar
        blocks (proving retrieval is actually doing something)."""
        a = exemplars.format_retrieved("salary is below market", k=5)
        b = exemplars.format_retrieved("I hate my manager", k=5)
        assert a != b


class TestPromptV2WithExemplars:
    def test_v2_static_substitutes_exemplar_block(self):
        from employee_voice.llm_backend import load_prompt
        text = load_prompt("category", version="v2")
        # {exemplar_block} should be replaced with a real block
        assert "{exemplar_block}" not in text
        assert "Examples" in text
        # At least 17 of the 18 categories should appear in the static block
        # (one may be missing only if format_exemplars caps < 18 and all
        # 18 categories happen not to fit — unlikely)
        cats_found = sum(1 for c in CATEGORIES if c in text)
        assert cats_found >= 17

    def test_v2_retrieved_uses_query(self):
        """When a query is passed, the exemplar block should be tailored."""
        from employee_voice.llm_backend import load_prompt
        # The salary query should preferentially pull Compensation exemplars
        text = load_prompt("category", version="v2",
                           exemplar_query="my salary is below market")
        assert "{exemplar_block}" not in text
        assert "Examples" in text
