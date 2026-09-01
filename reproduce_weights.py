#!/usr/bin/env python3
"""Convenience entry point for reproducing the selected final submissions.

The implementation lives in ``work/reproduce_frozen_finals.py``. Keeping this
small wrapper at the repository root preserves the original command name while
making it portable across clones and working directories.
"""
from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "work" / "reproduce_frozen_finals.py"


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
