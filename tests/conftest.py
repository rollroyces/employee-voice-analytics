"""Pytest configuration: register custom markers, set up sys.path."""
from __future__ import annotations
import os
import sys

# Ensure the in-repo package is importable when tests are run before install.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
