"""
`python -m employee_voice` entry point.

Lets you run the package as a module without installing the
console script. Equivalent to `eva` but uses the in-package argparse
shim. The Typer/Rich `eva` command is still the recommended CLI.
"""
from __future__ import annotations
import sys

# Thin wrapper around the legacy argparse CLI so the package has
# a `__main__` entry. The Typer-based `eva` console script is the
# primary CLI surface; this is here for `python -m employee_voice ...`.
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
