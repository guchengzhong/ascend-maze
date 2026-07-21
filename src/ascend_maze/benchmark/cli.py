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
    run = commands.add_parser("run", help="execute a frozen Study plan")
    run.add_argument("spec")
    run.add_argument("--output-root", default="experiment_output")
    resume = commands.add_parser("resume", help="resume an interrupted Study")
    resume.add_argument("study_directory")
    validate = commands.add_parser(
        "validate", help="import and validate a completed Study"
    )
    validate.add_argument("study_directory")
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
        if args.command in {"run", "resume"}:
            result = _run_or_resume(args)
            json.dump(
                result,
                sys.stdout,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            sys.stdout.write("\n")
            return 0 if result.get("state") == "completed" else 1
        if args.command == "validate":
            from ascend_maze.benchmark.importer import validate_study

            result = validate_study(args.study_directory)
            json.dump(
                result,
                sys.stdout,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            sys.stdout.write("\n")
            return 0 if result.get("study_valid") is True else 1
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
    except (TimeoutError, OSError, RuntimeError) as exc:
        json.dump(
            {
                "schema_version": 1,
                "status": "error",
                "error_code": "experiment_execution_failed",
                "message": str(exc),
            },
            sys.stderr,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        sys.stderr.write("\n")
        return 1


def _run_or_resume(args: argparse.Namespace) -> dict[str, object]:
    import asyncio

    from ascend_maze.benchmark.c13_runtime import C13BenchmarkRuntimeFactory
    from ascend_maze.benchmark.orchestrator import resume_study, run_study

    factory = C13BenchmarkRuntimeFactory()
    if args.command == "run":
        result = asyncio.run(
            run_study(
                args.spec,
                runtime_factory=factory,
                output_root=args.output_root,
            )
        )
    else:
        result = asyncio.run(
            resume_study(args.study_directory, runtime_factory=factory)
        )
    return result.canonical_payload()


if __name__ == "__main__":
    raise SystemExit(main())
