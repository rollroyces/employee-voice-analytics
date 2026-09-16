"""Layer-enforcement tests: dependencies, ordering, schema contracts."""
from __future__ import annotations
import os
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# Ensure the lightweight engines are allowed. conftest.py already sets
# this env var; we re-assert here because the layer tests poke at the
# ALLOW_FALLBACK global via set_allow_fallback() and need a known start.
os.environ["EMPLOYEE_VOICE_ALLOW_FALLBACK"] = "1"

# Importing the package triggers analyzers._detect_backends() which now
# raises if transformers is missing and ALLOW_FALLBACK is off. Make sure
# the env is set BEFORE we import the package.
import employee_voice.config as cfg  # noqa: E402
cfg.ALLOW_FALLBACK = True
from employee_voice import layers, pipeline  # noqa: E402


# ---------- dependency probe --------------------------------------------

class TestLayerDependencyProbe:
    def test_layer_contracts_are_sequential_0_through_6(self):
        layers_ = [c.layer for c in cfg.LAYER_CONTRACTS]
        assert layers_ == [0, 1, 2, 3, 4, 5, 6]

    def test_each_layer_contract_has_at_least_one_required_column_or_is_optional(self):
        for c in cfg.LAYER_CONTRACTS:
            assert c.required_columns or c.layer == 6, (
                f"Layer {c.layer} ({c.name}) declares no required columns; "
                f"only Layer 6 (llm_summary) is allowed to be column-less."
            )

    def test_missing_transformers_raises(self):
        """When transformers is unavailable and ALLOW_FALLBACK is off,
        the dependency probe must raise LayerDependencyError."""
        layers.set_allow_fallback(False)
        layer = cfg.LAYER_CONTRACTS[1]  # sentiment
        with mock.patch.dict(sys.modules, {"transformers": None}):
            with pytest.raises(layers.LayerDependencyError) as ei:
                layers._require_deps(layer)
        msg = str(ei.value)
        assert "transformers" in msg
        assert "pip install" in msg

    def test_missing_bertopic_raises(self):
        layers.set_allow_fallback(False)
        layer = cfg.LAYER_CONTRACTS[4]  # themes
        # Patch out the deps; require_deps only checks if import fails.
        with mock.patch.dict(sys.modules, {
            "bertopic": None,
            "sentence_transformers": None,
            "umap": None,
        }):
            with pytest.raises(layers.LayerDependencyError):
                layers._require_deps(layer)

    def test_missing_dep_with_fallback_on_is_quiet(self):
        layers.set_allow_fallback(True)
        layer = cfg.LAYER_CONTRACTS[1]
        with mock.patch.dict(sys.modules, {"transformers": None}):
            # Should not raise; should log instead.
            layers._require_deps(layer)
        # Reset for subsequent tests
        layers.set_allow_fallback(False)


# ---------- per-layer runners ------------------------------------------

class TestLayerRunners:
    """The individual layer functions populate the columns their contract
    requires, with sensible defaults for non-analyzable rows."""

    def _df(self):
        return pd.DataFrame({
            "FeedbackID": ["F1", "F2", "F3"],
            "Comment": [
                "I love working here, the team is amazing",
                "no comment",
                "Burned out, looking for a new opportunity",
            ],
        })

    def test_preprocess_layer(self):
        df, is_na, is_short = layers.run_preprocess(self._df())
        assert "CleanComment" in df.columns
        assert "Comment" in df.columns
        assert int((~is_na & ~is_short).sum()) == 2  # 2 analyzable

    def test_layer_1_sentiment(self):
        df, is_na, is_short = layers.run_preprocess(self._df())
        mask = (~is_na) & (~is_short)
        df = layers.run_sentiment(df, mask)
        assert {"Sentiment", "SentimentScore"}.issubset(df.columns)
        assert (df.loc[mask, "Sentiment"] != "No Comment").all()
        assert (df.loc[~mask, "Sentiment"] == "No Comment").all()

    def test_layer_5_risk_score(self):
        df, is_na, is_short = layers.run_preprocess(self._df())
        mask = (~is_na) & (~is_short)
        df = layers.run_sentiment(df, mask)
        df = layers.run_category(df, mask)
        df = layers.run_emotion(df, mask)
        df = layers.run_risk(df, mask)
        assert {"RiskScore", "RiskBand"}.issubset(df.columns)
        assert df["RiskScore"].between(0, 100).all()
        assert set(df["RiskBand"].unique()).issubset({"High", "Medium", "Low"})


# ---------- run_all_layers orchestrator ---------------------------------

class TestRunAllLayers:
    def _df(self, n: int = 10):
        rows = []
        for i in range(n):
            comment = (
                "I love working here, the team is amazing"
                if i % 3 == 0
                else "Burned out, looking for a new opportunity, salary is below market"
                if i % 3 == 1
                else "no comment"
            )
            rows.append({"FeedbackID": f"F{i+1:03d}", "Comment": comment})
        return pd.DataFrame(rows)

    def test_with_fallback_runs_end_to_end(self, tmp_path):
        layers.set_allow_fallback(True)
        try:
            # run_topics=False keeps the test fast and avoids TF-IDF
            # min_df constraints on tiny corpora.
            result = layers.run_all_layers(self._df(20), run_topics=False)
        finally:
            layers.set_allow_fallback(False)
        df = result["df"]
        for c in cfg.LAYER_CONTRACTS:
            if c.layer == 6:
                continue
            for col in c.required_columns:
                assert col in df.columns, f"Layer {c.layer} missing {col}"

    def test_contract_violation_raises(self):
        """If a runner forgets to populate a required column, the
        orchestrator must raise (LayerContractError or LayerOrderError
        depending on when the gap is detected) — never silently write
        nulls."""
        layers.set_allow_fallback(True)
        try:
            with mock.patch.object(layers, "run_sentiment",
                                   side_effect=lambda df, mask, **kw: df):  # noqa: ARG005
                with pytest.raises((layers.LayerContractError, layers.LayerOrderError)) as ei:
                    layers.run_all_layers(self._df(10))
            assert ("Sentiment" in str(ei.value)
                    or "Category1" in str(ei.value))
        finally:
            layers.set_allow_fallback(False)

    def test_layer_ordering_enforced(self):
        """The orchestrator must call layers in a fixed order and assert
        the upstream column exists before invoking the downstream layer.
        """
        layers.set_allow_fallback(True)
        try:
            with mock.patch.object(layers, "run_category",
                                   side_effect=lambda df, mask, **kw:
                                       df.drop(columns=["Category1"])
                                       if "Category1" in df.columns else df):
                with pytest.raises((layers.LayerContractError, layers.LayerOrderError)):
                    layers.run_all_layers(self._df(10))
        finally:
            layers.set_allow_fallback(False)


# ---------- end-to-end pipeline integration ----------------------------

class TestPipelineWithEnforcement:
    def test_pipeline_uses_layers_module(self):
        """The pipeline must delegate to layers.run_all_layers, not
        inline analyzer calls. This is a structural test — if someone
        regresses to inline calls, this fails."""
        import inspect
        from employee_voice import pipeline as p
        src = inspect.getsource(p.run)
        assert "run_all_layers" in src
        # And it must NOT call the analyzer funcs directly anymore.
        assert "analyze_sentiment(" not in src
        assert "discover_topics(" not in src
