"""Backwards-compatible shim. Prefer `python -m employee_voice` or the
installed `employee-voice` console script."""
from employee_voice.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
