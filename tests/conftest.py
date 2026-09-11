"""Pytest configuration: register custom markers, set up sys.path, enable
fallback engines for the unit / integration tests.

The Databricks notebook test (marked @pytest.mark.databricks) does its own
backend selection based on what is installed in the venv.
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

# Tests run with the lightweight engines unless a layer explicitly opts in.
os.environ.setdefault("EMPLOYEE_VOICE_ALLOW_FALLBACK", "1")


@pytest.fixture(autouse=True)
def _reset_layer_state():
    """Ensure every test starts with fallback engines enabled and a clean
    backend cache. Tests that want to assert the strict path can flip
    the flag inside their own body and re-flip on exit.
    """
    from employee_voice import config as _cfg
    _cfg.ALLOW_FALLBACK = True
    try:
        from employee_voice import analyzers as _an
        _an._reset_backends()
    except Exception:
        pass
    yield
    # Restore on exit so subsequent test files start clean.
    _cfg.ALLOW_FALLBACK = True
    try:
        from employee_voice import analyzers as _an
        _an._reset_backends()
    except Exception:
        pass
