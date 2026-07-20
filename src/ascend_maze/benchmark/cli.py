"""Offline-only ``maze-bench`` command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from ascend_maze import __version__
from ascend_maze.benchmark.loader import load_study_plan
from ascend_maze.core.errors import ContractValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maze-bench",
        description="Ascend-Maze deterministic experiment planner",
    )
    parser.add_argument("--version", action="store_true")
    commands = parser.add_subparsers(dest="command")
    plan = commands.add_parser("plan", help="validate and expand an ExperimentSpec")
    plan.add_argument("spec")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.version:
            print(f"Ascend-Maze benchmark {__version__} schema=1")
            return 0
        if args.command == "plan":
            plan = load_study_plan(args.spec)
            sys.stdout.write(plan.canonical_bytes.decode("utf-8") + "\n")
            return 0
        parser.print_help(sys.stderr)
        return 2
    except ContractValidationError as exc:
        json.dump(
            {
                "schema_version": 1,
                "status": "error",
                "error_code": "experiment_validation_failed",
                "message": str(exc),
            },
            sys.stderr,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        sys.stderr.write("\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
