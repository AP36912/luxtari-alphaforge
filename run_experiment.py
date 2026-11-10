#!/usr/bin/env python3
"""Run a configured experiment or deterministic public synthetic demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.yaml_loader import load_experiment_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Luxtari AlphaForge experiment runner")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Experiment YAML path")
    source.add_argument("--synthetic", action="store_true", help="Run seeded offline SMA demo")
    parser.add_argument("--validate-only", action="store_true", help="Load and summarize without running")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=750)
    args = parser.parse_args()

    if args.synthetic:
        if args.validate_only:
            print(json.dumps({"mode": "synthetic", "seed": args.seed, "rows": args.rows,
                              "network": False, "broker": False}, indent=2))
            return
        from research.public_offline_run import run_public_synthetic
        report = run_public_synthetic(seed=args.seed, rows=args.rows)
        print(json.dumps({"report": str(report)}, indent=2))
        return

    assert args.config is not None
    cfg = load_experiment_yaml(args.config)
    summary = {"type": type(cfg).__name__, "source": str(args.config),
               "execution_mode": cfg.execution.execution_mode, "seed": cfg.seed}
    if args.validate_only:
        print(json.dumps(summary, indent=2, default=str))
        return
    from config.pipeline import run_experiment
    result = run_experiment(cfg)
    print(json.dumps({**summary, "run_dir": str(result["run_dir"]),
                      "metrics_path": str(result["metrics_path"])}, indent=2))


if __name__ == "__main__":
    main()
