#!/usr/bin/env python3
"""Prove concurrent Worker dispatch ownership through a live Ray Controller."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

from ascend_maze import Workflow
from ascend_maze.control.local_rpc import UdsRuntimeClient
from ascend_maze.experiments.async_dispatch_probe import cold_dispatch_probe


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_workflow() -> tuple[Workflow, str]:
    workflow = Workflow("async-cold-dispatch-probe")
    probe_id = workflow.input("probe_id")
    node = workflow.add_task(
        cold_dispatch_probe,
        inputs={"probe_id": probe_id, "hold_seconds": 0.5},
    )
    return workflow, node.task_id


def _events(batches: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for batch in batches:
        values = batch.get("events")
        if not isinstance(values, list):
            continue
        events.extend(item for item in values if isinstance(item, dict))
    return events


def _active_worker_leases(snapshot: Mapping[str, object]) -> list[object]:
    pool = snapshot.get("worker_pool")
    if not isinstance(pool, dict):
        return []
    leases = pool.get("worker_leases")
    if not isinstance(leases, list):
        return []
    return [
        item
        for item in leases
        if isinstance(item, dict) and not bool(item.get("released"))
    ]


def _live_workers(snapshot: Mapping[str, object]) -> list[object]:
    pool = snapshot.get("worker_pool")
    if not isinstance(pool, dict):
        return []
    workers = pool.get("workers")
    if not isinstance(workers, list):
        return []
    return [
        item
        for item in workers
        if isinstance(item, dict) and item.get("state") != "dead"
    ]


def _worker_pool(snapshot: Mapping[str, object]) -> dict[str, object]:
    pool = snapshot.get("worker_pool")
    return dict(pool) if isinstance(pool, dict) else {}


def _live_worker_signature(snapshot: Mapping[str, object]) -> list[str]:
    return sorted(
        json.dumps(
            {
                "node_id": item.get("node_id"),
                "boot_id": item.get("boot_id"),
                "profile": item.get("profile"),
                "state": item.get("state"),
            },
            sort_keys=True,
        )
        for item in _live_workers(snapshot)
        if isinstance(item, dict)
    )


def _global_lease_signature(snapshot: Mapping[str, object]) -> list[str]:
    cluster = snapshot.get("cluster")
    if not isinstance(cluster, dict):
        return []
    leases = cluster.get("active_leases")
    if not isinstance(leases, list):
        return []
    return sorted(
        json.dumps(
            {
                "reservation_kind": lease.get("reservation_kind"),
                "node_id": lease.get("node_id"),
                "boot_id": lease.get("boot_id"),
                "npu_device_id": lease.get("npu_device_id"),
                "resources": lease.get("resources"),
            },
            sort_keys=True,
        )
        for item in leases
        if isinstance(item, dict)
        for lease in [item.get("lease")]
        if isinstance(lease, dict) and lease.get("run_id") is None
    )


def _hbm_free_by_device(snapshot: Mapping[str, object]) -> dict[str, int]:
    cluster = snapshot.get("cluster")
    if not isinstance(cluster, dict):
        return {}
    nodes = cluster.get("nodes")
    if not isinstance(nodes, list):
        return {}
    result: dict[str, int] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        capacity = node.get("capacity")
        if not isinstance(capacity, dict):
            continue
        node_id = capacity.get("node_id")
        npus = capacity.get("npus")
        if not isinstance(node_id, str) or not isinstance(npus, list):
            continue
        for npu in npus:
            if not isinstance(npu, dict):
                continue
            device_id = npu.get("device_id")
            free_hbm = npu.get("observed_free_hbm_mb")
            if isinstance(device_id, str) and isinstance(free_hbm, int):
                result[f"{node_id}:{device_id}"] = free_hbm
    return result


def _pool_failure_counters(snapshot: Mapping[str, object]) -> dict[str, int]:
    pool = _worker_pool(snapshot)
    return {
        name: int(pool.get(name, 0))
        for name in (
            "replenish_failures",
            "sanitize_failures",
            "termination_failures",
        )
    }


def _run_leases(snapshot: Mapping[str, object], run_ids: set[str]) -> list[object]:
    cluster = snapshot.get("cluster")
    if not isinstance(cluster, dict):
        return []
    leases = cluster.get("active_leases")
    if not isinstance(leases, list):
        return []
    result: list[object] = []
    for item in leases:
        if not isinstance(item, dict):
            continue
        lease = item.get("lease")
        if isinstance(lease, dict) and lease.get("run_id") in run_ids:
            result.append(item)
    return result


def _attempt_violations(runs: Sequence[object]) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for run in runs:
        if not isinstance(run, dict):
            violations.append({"reason": "run_snapshot_missing"})
            continue
        for task in run.get("task_states", []):
            if not isinstance(task, dict):
                continue
            attempts = task.get("attempts")
            valid_attempts = attempts if isinstance(attempts, list) else []
            valid = (
                task.get("status") == "succeeded"
                and task.get("attempt_count") == 1
                and len(valid_attempts) == 1
                and isinstance(valid_attempts[0], dict)
                and valid_attempts[0].get("attempt") == 1
                and valid_attempts[0].get("status") == "succeeded"
            )
            if not valid:
                violations.append(
                    {
                        "run_id": run.get("run_id"),
                        "task_id": task.get("task_id"),
                        "status": task.get("status"),
                        "attempt_count": task.get("attempt_count"),
                        "attempts": valid_attempts,
                    }
                )
    return violations


async def _watch(
    client: UdsRuntimeClient,
    run_id: str,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    return [
        batch
        async for batch in client.watch_run(
            run_id,
            after_sequence=0,
            limit=500,
            timeout_seconds=timeout_seconds,
        )
    ]


async def _wait_cleanup(
    client: UdsRuntimeClient,
    run_ids: set[str],
    baseline_workers: Mapping[str, object],
    baseline_cluster: Mapping[str, object],
    timeout_seconds: float,
    hbm_tolerance_mb: int = 512,
) -> dict[str, object]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last: dict[str, object] = {}
    baseline_live_workers = _live_worker_signature(baseline_workers)
    baseline_global_leases = _global_lease_signature(baseline_cluster)
    baseline_hbm = _hbm_free_by_device(baseline_cluster)
    baseline_failures = _pool_failure_counters(baseline_workers)
    while asyncio.get_running_loop().time() < deadline:
        workers, cluster = await asyncio.gather(
            client.query("GetWorkerPools", timeout_seconds=10),
            client.query(
                "GetClusterSnapshot",
                filter="resources",
                timeout_seconds=10,
            ),
        )
        current_live_workers = _live_worker_signature(workers)
        current_global_leases = _global_lease_signature(cluster)
        current_hbm = _hbm_free_by_device(cluster)
        current_failures = _pool_failure_counters(workers)
        hbm_recovered = all(
            current_hbm.get(device_id, -1) >= free_hbm - hbm_tolerance_mb
            for device_id, free_hbm in baseline_hbm.items()
        )
        last = {
            "workers": workers,
            "cluster": cluster,
            "baseline_live_worker_signature": baseline_live_workers,
            "current_live_worker_signature": current_live_workers,
            "baseline_global_lease_signature": baseline_global_leases,
            "current_global_lease_signature": current_global_leases,
            "baseline_hbm_free_mb": baseline_hbm,
            "current_hbm_free_mb": current_hbm,
            "hbm_tolerance_mb": hbm_tolerance_mb,
            "hbm_recovered": hbm_recovered,
            "baseline_pool_failure_counters": baseline_failures,
            "current_pool_failure_counters": current_failures,
        }
        if (
            not _active_worker_leases(workers)
            and not _run_leases(cluster, run_ids)
            and current_live_workers == baseline_live_workers
            and current_global_leases == baseline_global_leases
            and current_failures == baseline_failures
            and hbm_recovered
        ):
            last["recovered"] = True
            return last
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Worker dispatch resources did not recover: {last}")


async def _wait_initial_baseline(
    client: UdsRuntimeClient,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, object], dict[str, object]]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last: tuple[dict[str, object], dict[str, object]] | None = None
    while asyncio.get_running_loop().time() < deadline:
        workers, cluster = await asyncio.gather(
            client.query("GetWorkerPools", timeout_seconds=10),
            client.query(
                "GetClusterSnapshot",
                filter="resources",
                timeout_seconds=10,
            ),
        )
        last = (workers, cluster)
        pool = _worker_pool(workers)
        mode = pool.get("mode")
        live_workers = _live_workers(workers)
        all_idle = all(
            isinstance(worker, dict) and worker.get("state") == "idle"
            for worker in live_workers
        )
        expected_count = 0 if mode == "cold_start" else 24
        if (
            mode in {"cold_start", "zero_hbm_standby"}
            and not _active_worker_leases(workers)
            and len(live_workers) == expected_count
            and all_idle
        ):
            return workers, cluster
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Worker Pool did not reach its idle baseline: {last}")


async def _run_batch(
    client: UdsRuntimeClient,
    *,
    batch_size: int,
    worker_pool_mode: str,
    baseline_workers: Mapping[str, object],
    baseline_cluster: Mapping[str, object],
    timeout_seconds: float,
) -> dict[str, object]:
    workflow, task_id = _build_workflow()
    compiled = workflow.compile()
    nonce = time.time_ns()
    prepared = []
    for index in range(batch_size):
        identity = f"async-dispatch:{batch_size}:{index}:{nonce}"
        submission_id = "async-" + hashlib.sha256(identity.encode()).hexdigest()[:28]
        prepared.append(
            await client.prepare_submission(
                workflow,
                inputs={"probe_id": identity},
                submission_id=submission_id,
                run_deadline_ms=round(timeout_seconds * 1_000),
            )
        )
    outcomes = await asyncio.gather(
        *(client.submit_prepared(item, timeout_seconds=30) for item in prepared)
    )
    run_ids = {
        str(outcome["run_id"])
        for outcome in outcomes
        if isinstance(outcome.get("run_id"), str)
    }
    if len(run_ids) != batch_size:
        raise RuntimeError(f"not every submission committed: {outcomes}")

    watch_by_run = dict(
        zip(
            sorted(run_ids),
            await asyncio.gather(
                *(_watch(client, run_id, timeout_seconds) for run_id in sorted(run_ids))
            ),
            strict=True,
        )
    )
    shown = await asyncio.gather(
        *(
            client.query("GetRun", resource_id=run_id, timeout_seconds=10)
            for run_id in sorted(run_ids)
        )
    )
    runs = [item.get("run") for item in shown]
    if any(
        not isinstance(run, dict) or run.get("status") != "succeeded"
        for run in runs
    ):
        raise RuntimeError(f"one or more probe Runs failed: {runs}")
    attempt_violations = _attempt_violations(runs)
    if attempt_violations:
        raise RuntimeError(f"successful Tasks did not stay on Attempt 1: {attempt_violations}")

    results = await asyncio.gather(
        *(client.materialize_task_result(run_id, task_id) for run_id in sorted(run_ids))
    )
    events = [
        event
        for run_id in sorted(run_ids)
        for event in _events(watch_by_run[run_id])
    ]
    requested = [item for item in events if item.get("event_type") == "task_dispatched"]
    prepared_events = [
        item for item in events if item.get("event_type") == "dispatch_prepared"
    ]
    worker_started = [
        item for item in events if item.get("event_type") == "worker_started"
    ]
    dispatch_start_failed = [
        item for item in events if item.get("event_type") == "dispatch_start_failed"
    ]
    if dispatch_start_failed:
        raise RuntimeError(f"dispatch startup failed internally: {dispatch_start_failed}")
    if not (
        len(requested) == len(prepared_events) == len(worker_started) == batch_size
    ):
        raise RuntimeError(
            "missing dispatch lifecycle events: "
            f"requested={len(requested)} prepared={len(prepared_events)} "
            f"worker_started={len(worker_started)}"
        )
    requested_sequences = [int(item["sequence"]) for item in requested]
    prepared_sequences = [int(item["sequence"]) for item in prepared_events]
    all_requested_before_first_prepared = (
        max(requested_sequences) < min(prepared_sequences)
    )
    if worker_pool_mode == "cold_start" and not all_requested_before_first_prepared:
        raise RuntimeError(
            "cold dispatch startups did not overlap: "
            f"requested={requested_sequences}, prepared={prepared_sequences}"
        )

    node_ids = sorted(
        {
            str(attempt["node_id"])
            for run in runs
            if isinstance(run, dict)
            for task in run.get("task_states", [])
            if isinstance(task, dict)
            for attempt in task.get("attempts", [])
            if isinstance(attempt, dict) and isinstance(attempt.get("node_id"), str)
        }
    )
    if len(node_ids) != batch_size:
        raise RuntimeError(
            f"expected {batch_size} logical task nodes, observed {node_ids}"
        )

    destroy_results = await asyncio.gather(
        *(
            client.run_action(
                "DestroyRun",
                run_id,
                force=True,
                timeout_seconds=120,
            )
            for run_id in sorted(run_ids)
        )
    )
    cleanup = await _wait_cleanup(
        client,
        run_ids,
        baseline_workers,
        baseline_cluster,
        timeout_seconds=30,
    )
    return {
        "batch_size": batch_size,
        "run_ids": sorted(run_ids),
        "workflow_fingerprint": compiled.workflow_fingerprint,
        "task_id": task_id,
        "node_ids": node_ids,
        "requested_sequences": requested_sequences,
        "prepared_sequences": prepared_sequences,
        "worker_started_sequences": [int(item["sequence"]) for item in worker_started],
        "all_requested_before_first_prepared": all_requested_before_first_prepared,
        "worker_pool_mode": worker_pool_mode,
        "dispatch_start_failed_count": len(dispatch_start_failed),
        "attempt_violations": attempt_violations,
        "results": results,
        "runs": runs,
        "events": sorted(events, key=lambda item: int(item["sequence"])),
        "destroy_results": destroy_results,
        "cleanup": cleanup,
        "succeeded": True,
    }


async def run(args: argparse.Namespace) -> int:
    client = UdsRuntimeClient(args.control_socket.expanduser().resolve())
    output = args.output.expanduser().resolve()
    try:
        status = await client.get_controller_status(timeout_seconds=10)
        if status.healthy_node_count != 8:
            raise RuntimeError(
                f"expected eight healthy logical nodes, found {status.healthy_node_count}"
            )
        workers, cluster = await _wait_initial_baseline(client, timeout_seconds=30)
        pool = _worker_pool(workers)
        worker_pool_mode = str(pool.get("mode"))
        await client._ensure_data_store()  # noqa: SLF001
        results = [
            await _run_batch(
                client,
                batch_size=size,
                worker_pool_mode=worker_pool_mode,
                baseline_workers=workers,
                baseline_cluster=cluster,
                timeout_seconds=args.timeout_seconds,
            )
            for size in args.batch_size
        ]
        summary = {
            "schema_version": 1,
            "objective": "async_cold_dispatch_acceptance",
            "controller_status": {
                "controller_generation": status.controller_generation,
                "build_revision": status.build_revision,
                "environment_fingerprint": status.environment_fingerprint,
                "healthy_node_count": status.healthy_node_count,
            },
            "initial_worker_pool": workers,
            "initial_cluster": cluster,
            "results": results,
            "succeeded": all(bool(item.get("succeeded")) for item in results),
        }
        _write_json(output, summary)
        print(json.dumps({"succeeded": summary["succeeded"], "output": str(output)}))
        return 0 if summary["succeeded"] else 1
    finally:
        client.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-socket",
        type=Path,
        default=Path("/workspace/state/control-plane/control.sock"),
    )
    parser.add_argument("--batch-size", action="append", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/state/output/async-dispatch-acceptance.json"),
    )
    args = parser.parse_args(argv)
    if args.batch_size is None:
        args.batch_size = [2, 4]
    if any(item < 2 or item > 8 for item in args.batch_size):
        parser.error("--batch-size must be within 2..8")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
