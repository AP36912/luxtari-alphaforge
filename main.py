"""
Primary CLI entry for Luxtari AlphaForge.

Later phases will route subcommands (e.g. ingest, features, train, backtest)
through this module while keeping business logic in subpackages.
"""

from __future__ import annotations


def main() -> None:
    """
    Placeholder entry point.

    Financial reasoning: orchestration stays thin so research logic remains
    testable in isolation (avoids hiding lookahead or cost assumptions in CLI glue).
    """
    # Phase 1: skeleton only. Wire argparse / Typer in a later phase.
    print(
        "luxtari-alphaforge: Phase 1 skeleton. "
        "Use run_experiment.py and package imports in later phases."
    )


if __name__ == "__main__":
    main()
