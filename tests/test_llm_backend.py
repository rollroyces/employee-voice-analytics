"""Tests for the hosted-LLM backend.

These tests do NOT hit any real LLM API. They exercise the
plumbing (provider detection, prompt loading, batching, JSON
parsing, PII scrubbing, error messages). The mocked-call path is
the only path CI runs.
"""
from __future__ import annotations
import json
import os
from typing import List
from unittest.mock import patch

import pandas as pd
import pytest

from employee_voice import llm_backend
from employee_voice.llm_backend import (
    Provider,
    detect_provider,
    load_prompt,
    batched_classify,
    _parse_response,
    chunk,
)


class TestProviderDetection:
    def setup_method(self):
        # Snapshot env so each test can mutate it.
        self._saved = {
            k: os.environ.get(k, "")
            for k in (
                "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION",
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
            )
        }
        for k in self._saved:
            os.environ.pop(k, None)

    def teardown_method(self):
        for k, v in self._saved.items():
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    def test_returns_none_with_no_env(self):
        assert detect_provider() is None

    def test_azure_wins_over_openai(self):
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://x.openai.azure.com"
        os.environ["AZURE_OPENAI_API_KEY"] = "key"
        os.environ["OPENAI_API_KEY"] = "openai-key"
        p = detect_provider()
        assert p is not None
        assert p.name == "azure"

    def test_deployment_default(self):
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://x.openai.azure.com"
        os.environ["AZURE_OPENAI_API_KEY"] = "key"
        p = detect_provider()
        assert p is not None
        assert p.deployment == "gpt-4o-mini"

    def test_deployment_override(self):
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://x.openai.azure.com"
        os.environ["AZURE_OPENAI_API_KEY"] = "key"
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o"
        p = detect_provider()
        assert p is not None and p.deployment == "gpt-4o"


class TestPromptLoading:
    def test_load_category_v1(self):
        text = load_prompt("category")
        # The prompt mentions the categories by name
        assert "Compensation and Benefits" in text
        assert "Work-Life Balance" in text

    def test_load_sentiment_v1(self):
        text = load_prompt("sentiment")
        assert "positive" in text
        assert "negative" in text

    def test_load_emotion_v1(self):
        text = load_prompt("emotion")
        assert "gratitude" in text
        assert "anger" in text

    def test_fallback_to_v1_when_version_missing(self):
        # Asking for v999 should fall back to v1 with a warning.
        text = load_prompt("category", version="v999")
        assert "Compensation and Benefits" in text

    def test_unknown_task_raises(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_task")


class TestChunk:
    def test_chunks_evenly(self):
        out = list(chunk(["a", "b", "c", "d"], 2))
        assert out == [["a", "b"], ["c", "d"]]

    def test_chunks_with_remainder(self):
        out = list(chunk(["a", "b", "c", "d", "e"], 2))
        assert out == [["a", "b"], ["c", "d"], ["e"]]

    def test_empty_input(self):
        out = list(chunk([], 3))
        assert out == []

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            list(chunk(["a"], 0))


class TestParseResponse:
    def test_clean_object(self):
        body = json.dumps({"results": [{"label": "positive", "score": 0.9},
                                       {"label": "negative", "score": 0.1}]})
        rows = _parse_response(body, expected_n=2, layer="sentiment")
        assert rows == [{"label": "positive", "score": 0.9},
                        {"label": "negative", "score": 0.1}]

    def test_code_fenced(self):
        body = (
            "```json\n"
            + json.dumps({"results": [{"label": "positive", "score": 0.9}]})
            + "\n```"
        )
        rows = _parse_response(body, expected_n=1, layer="sentiment")
        assert rows[0]["label"] == "positive"

    def test_bare_list(self):
        body = json.dumps([{"label": "x", "score": 0.5}])
        rows = _parse_response(body, expected_n=1, layer="sentiment")
        assert rows[0]["label"] == "x"

    def test_wrong_count_raises(self):
        body = json.dumps({"results": [{"label": "x", "score": 0.1}]})
        with pytest.raises(ValueError, match="expected 3"):
            _parse_response(body, expected_n=3, layer="sentiment")

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="no JSON"):
            _parse_response("just some text", expected_n=1, layer="sentiment")


class TestBatchedClassify:
    def setup_method(self):
        # No real provider in tests.
        self._saved = {
            k: os.environ.get(k, "")
            for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
                      "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        }
        for k in self._saved:
            os.environ.pop(k, None)

    def teardown_method(self):
        for k, v in self._saved.items():
            if v:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    def test_empty_rows_returns_empty(self):
        provider = Provider(name="openai", deployment="gpt-4o-mini")
        os.environ["OPENAI_API_KEY"] = "fake"
        with patch.object(llm_backend, "_call_openai_compatible",
                          return_value=json.dumps({"results": []})):
            out = batched_classify([], task="sentiment", provider=provider)
            assert out == []

    def test_no_provider_raises(self):
        with pytest.raises(RuntimeError, match="no LLM provider"):
            batched_classify(["hello"], task="sentiment")

    def test_unknown_task_raises(self):
        provider = Provider(name="openai", deployment="gpt-4o-mini")
        os.environ["OPENAI_API_KEY"] = "fake"
        with pytest.raises(ValueError, match="unknown task"):
            batched_classify(["hello"], task="bogus", provider=provider)

    def test_pii_scrub_default(self):
        """Emails and phones should be redacted before they ever
        reach the (mocked) HTTP call."""
        provider = Provider(name="openai", deployment="gpt-4o-mini")
        os.environ["OPENAI_API_KEY"] = "fake"

        captured_text: List[str] = []

        def fake_call(p, system, rows, *args, **kwargs):
            captured_text.extend(rows)
            return json.dumps({"results": [{"label": "neutral", "score": 0.5}] * len(rows)})

        with patch.object(llm_backend, "_call_openai_compatible", side_effect=fake_call):
            batched_classify(
                ["contact me at alice@example.com for details"],
                task="sentiment",
                provider=provider,
            )
        # The scrubber should have removed the email
        assert "alice@example.com" not in captured_text[0]
        # But preserved the surrounding text
        assert "contact me at" in captured_text[0]

    def test_send_raw_text_keeps_pii(self):
        provider = Provider(name="openai", deployment="gpt-4o-mini")
        os.environ["OPENAI_API_KEY"] = "fake"

        captured_text: List[str] = []

        def fake_call(p, system, rows, *args, **kwargs):
            captured_text.extend(rows)
            return json.dumps({"results": [{"label": "neutral", "score": 0.5}] * len(rows)})

        with patch.object(llm_backend, "_call_openai_compatible", side_effect=fake_call):
            batched_classify(
                ["contact me at alice@example.com"],
                task="sentiment",
                provider=provider,
                send_raw_text=True,
            )
        # send_raw_text=True -> no scrubbing
        assert "alice@example.com" in captured_text[0]

    def test_offset_preserved_when_response_truncated(self):
        """If the model returns malformed JSON, the error message
        identifies which batch failed."""
        provider = Provider(name="openai", deployment="gpt-4o-mini")
        os.environ["OPENAI_API_KEY"] = "fake"

        def bad_call(p, system, rows, *args, **kwargs):
            return "this is not json"

        with patch.object(llm_backend, "_call_openai_compatible", side_effect=bad_call):
            with pytest.raises(ValueError, match="sentiment"):
                batched_classify(["a", "b"], task="sentiment", provider=provider)

    def test_outputs_aligned_to_inputs(self):
        """The order of output dicts must match the order of inputs."""
        provider = Provider(name="openai", deployment="gpt-4o-mini")
        os.environ["OPENAI_API_KEY"] = "fake"

        # Track the global index of each row across all batches so
        # we can verify outputs map back to inputs.
        global_idx = [0]

        def echo_call(p, system, rows, *args, **kwargs):
            out = []
            for _ in rows:
                out.append({"label": f"l{global_idx[0]}", "score": 0.5})
                global_idx[0] += 1
            return json.dumps({"results": out})

        with patch.object(llm_backend, "_call_openai_compatible", side_effect=echo_call):
            out = batched_classify(
                ["x", "y", "z", "w"],
                task="sentiment",
                provider=provider,
                batch_size=2,
            )
        assert len(out) == 4
        assert [d["label"] for d in out] == ["l0", "l1", "l2", "l3"]

    def test_model_override(self):
        provider = Provider(name="openai", deployment="gpt-4o-mini")
        os.environ["OPENAI_API_KEY"] = "fake"

        seen_provider: List[Provider] = []

        def fake_call(p, system, rows, *args, **kwargs):
            seen_provider.append(p)
            return json.dumps({"results": [{"label": "x", "score": 0.5}] * len(rows)})

        with patch.object(llm_backend, "_call_openai_compatible", side_effect=fake_call):
            batched_classify(
                ["hi"], task="sentiment",
                provider=provider, model="gpt-4o",
            )
        assert seen_provider[0].deployment == "gpt-4o"


class TestAnalyzeCategoryLLMMocked:
    """Integration: analyze_category_llm with a mocked HTTP call.
    Verifies the DataFrame contract matches the HF path."""

    def test_returns_correct_columns(self):
        from employee_voice.analyzers import analyze_category_llm

        with patch.object(
            llm_backend, "batched_classify",
            return_value=[
                {"category": "Compensation and Benefits", "score": 0.9},
                {"category": "Management",
                 "secondary": "Workload", "score": 0.85},
            ],
        ):
            df = analyze_category_llm(["low pay", "bad manager + heavy workload"])
        assert list(df.columns) == [
            "Category1", "Category1Score", "Category2",
            "Category2Score", "AllCategories",
        ]
        assert len(df) == 2
        assert df.iloc[0]["Category1"] == "Compensation and Benefits"
        assert df.iloc[1]["Category1"] == "Management"
        assert df.iloc[1]["Category2"] == "Workload"

    def test_alias_normalisation(self):
        from employee_voice.analyzers import analyze_category_llm

        with patch.object(
            llm_backend, "batched_classify",
            return_value=[
                {"category": "salary", "score": 0.9},
                {"category": "wlb", "score": 0.8},
            ],
        ):
            df = analyze_category_llm(["x", "y"])
        assert df.iloc[0]["Category1"] == "Compensation and Benefits"
        assert df.iloc[1]["Category1"] == "Work-Life Balance"

    def test_unknown_category_falls_back_to_other(self):
        from employee_voice.analyzers import analyze_category_llm

        with patch.object(
            llm_backend, "batched_classify",
            return_value=[{"category": "teleportation", "score": 0.9}],
        ):
            df = analyze_category_llm(["x"])
        assert df.iloc[0]["Category1"] == "Other"

    def test_prompt_includes_taxonomy(self):
        """The category prompt must mention every entry in CATEGORIES
        so the LLM can't drift from the runtime taxonomy."""
        from employee_voice.config import CATEGORIES
        text = load_prompt("category")
        for c in CATEGORIES:
            assert c in text, f"category {c!r} missing from category prompt"


class TestAnalyzeSentimentLLMMocked:
    def test_returns_correct_columns(self):
        from employee_voice.analyzers import analyze_sentiment_llm

        with patch.object(
            llm_backend, "batched_classify",
            return_value=[
                {"label": "positive", "score": 0.9},
                {"label": "negative", "score": 0.85},
            ],
        ):
            df = analyze_sentiment_llm(["great", "terrible"])
        assert list(df.columns) == ["Sentiment", "SentimentScore"]
        assert df.iloc[0]["Sentiment"] == "positive"
        assert df.iloc[1]["Sentiment"] == "negative"
        assert df.iloc[0]["SentimentScore"] == 0.9


class TestAnalyzeEmotionLLMMocked:
    def test_returns_correct_columns(self):
        from employee_voice.analyzers import analyze_emotion_llm

        with patch.object(
            llm_backend, "batched_classify",
            return_value=[
                {"label": "gratitude", "score": 0.95},
                {"label": "anger", "score": 0.8},
            ],
        ):
            df = analyze_emotion_llm(["thanks!", "this is awful"])
        assert list(df.columns) == ["Emotion", "EmotionScore"]
        assert df.iloc[0]["Emotion"] == "gratitude"
        assert df.iloc[1]["Emotion"] == "anger"


class TestLayerBackendPlumbing:
    """Confirmed the run_*_layer functions accept the backend arg
    and route to the right function."""

    def test_run_sentiment_default_uses_analyze_sentiment(self):
        from employee_voice import layers as l
        df = pd.DataFrame({"Comment": ["great manager"], "CleanComment": ["great manager"]})
        mask = pd.Series([True])
        # ALLOW_FALLBACK should make the default path work
        from employee_voice import config as cfg
        old = cfg.ALLOW_FALLBACK
        try:
            cfg.set_allow_fallback(True)
            out = l.run_sentiment(df.copy(), mask, backend="auto")
            assert "Sentiment" in out.columns
            assert "SentimentScore" in out.columns
        finally:
            cfg.set_allow_fallback(old)

    def test_run_sentiment_llm_backend_routes_to_llm(self):
        from employee_voice import layers as l
        df = pd.DataFrame({"Comment": ["x"], "CleanComment": ["x"]})
        mask = pd.Series([True])
        with patch(
            "employee_voice.analyzers.analyze_sentiment_llm",
            return_value=pd.DataFrame({"Sentiment": ["positive"], "SentimentScore": [0.9]}),
        ) as m:
            out = l.run_sentiment(df.copy(), mask, backend="llm", llm_options={})
            m.assert_called_once()
            assert out.iloc[0]["Sentiment"] == "positive"
