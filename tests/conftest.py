"""Pytest configuration: register custom markers, set up sys.path.

The Databricks notebook test is marked @pytest.mark.databricks. It is
**deselected by default** via pyproject's `addopts = "-m 'not databricks'"`
so plain `pytest` works on machines without Java + PySpark. Run it
explicitly with `pytest -m databricks`.

The fallback-engines flag follows the env var: if the user has
explicitly set EMPLOYEE_VOICE_ALLOW_FALLBACK=0 we respect that. Only
when the env var is unset do we default it to "1" for test convenience.
"""
from __future__ import annotations
import os
import sys

import pytest

# Ensure the in-repo package is importable when tests are run before install.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

# Default to fallback engines for the test environment — but only if the
# user hasn't explicitly set the env var to "0" (which is how CI / a
# strict-mode smoke test would opt out).
os.environ.setdefault("EMPLOYEE_VOICE_ALLOW_FALLBACK", "1")


@pytest.fixture(autouse=True)
def _reset_layer_state():
    """Sync the global flag to the env var before each test, then reset
    the backend cache so the analyzer re-detects. This honours an
    explicit `EMPLOYEE_VOICE_ALLOW_FALLBACK=0` in the shell — it only
    forces fallback on when the env var is unset / "1".
    """
    from employee_voice import config as _cfg
    env_val = os.environ.get("EMPLOYEE_VOICE_ALLOW_FALLBACK", "1")
    _cfg.ALLOW_FALLBACK = env_val not in ("0", "false", "False", "")
    try:
        from employee_voice import analyzers as _an
        _an._reset_backends()
    except Exception:
        pass
    yield
    # Restore the post-test state to the env-var value so other test
    # files start in the same mode the user asked for.
    _cfg.ALLOW_FALLBACK = env_val not in ("0", "false", "False", "")
    try:
        from employee_voice import analyzers as _an
        _an._reset_backends()
    except Exception:
        pass
