#!/usr/bin/env python
"""Re-render the Markdown report from a previous run, without re-running any model.

Useful after changing a presentation-only setting such as
``output.error_examples_in_report`` (redacted vs full).

    python scripts/render_report.py
    python scripts/render_report.py --config my_experiment.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import reporting  # noqa: E402
from src.config import load_config  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-render the benchmark report")
    parser.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    summary_path = cfg.results_dir / "summary.json"
    if not summary_path.exists():
        print("no %s: run `python run_benchmark.py` first" % summary_path, file=sys.stderr)
        return 1
    with open(summary_path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    path = reporting.generate_report(summary, cfg)
    print("re-rendered: %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
