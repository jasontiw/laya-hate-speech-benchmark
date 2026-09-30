#!/usr/bin/env python
"""Thin wrapper so the benchmark can also be launched as ``scripts/run_benchmark.py``.

The real entry point lives at the project root (PRD section 19); this keeps the
``scripts/`` layout from section 25 working as well.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
