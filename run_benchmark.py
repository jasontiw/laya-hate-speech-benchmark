#!/usr/bin/env python
"""Entry point: reproduce the whole benchmark with one command.

    python run_benchmark.py                  # full run, config.yaml
    python run_benchmark.py --limit 200      # dev smoke run on 200 test tweets
    python run_benchmark.py --models tfidf,laya
    python run_benchmark.py --list-models
    python run_benchmark.py --config my.yaml

Steps (PRD section 19): validate environment -> load/verify dataset -> build split
-> load models -> predict -> metrics -> charts -> report -> save results.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CANONICAL_LABELS, load_config  # noqa: E402
from src.evaluation import run_benchmark  # noqa: E402
from src.models import available_model_keys  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Hate Speech Detection Benchmark")
    parser.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated subset of the configured model keys (see --list-models)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="DEBUG: evaluate only the first N test rows (not a benchmark result)",
    )
    parser.add_argument("--list-models", action="store_true", help="print the configured model keys and exit")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    keys = available_model_keys(cfg)

    if args.list_models:
        print("\n".join(keys))
        return 0

    only = None
    if args.models:
        only = [name.strip() for name in args.models.split(",") if name.strip()]
        unknown = [name for name in only if name not in keys]
        if unknown:
            print("unknown model(s): %s\nconfigured: %s" % (", ".join(unknown), ", ".join(keys)), file=sys.stderr)
            return 2

    print("=" * 78)
    print("  Local Hate Speech Detection Benchmark")
    print("  config: %s" % cfg.path)
    print("  labels: %s" % ", ".join(CANONICAL_LABELS))
    print("  model configurations: %d (%s)" % (len(keys), ", ".join(keys)))
    print("  bootstrap: %d resamples, seed %d" % (cfg.stats.bootstrap_samples, cfg.stats.seed))
    if args.limit:
        print("  MODE: DEBUG --limit %d (not a benchmark result)" % args.limit)
    print("=" * 78)

    started = time.perf_counter()
    summary = run_benchmark(cfg, only=only, limit=args.limit)
    elapsed = time.perf_counter() - started

    print("=" * 78)
    print("  done in %.1fs" % elapsed)
    print("  results : %s" % summary["results_dir"])
    print("  report  : %s" % (cfg.report_dir / "benchmark_report.md"))
    if summary.get("comparison"):
        print("  models  : %s" % ", ".join(row["model"] for row in summary["comparison"]))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
