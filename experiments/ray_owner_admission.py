"""Stage 3A admission test for Ray's long-lived object owner mechanism."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable
import uuid

import cloudpickle
import ray
from ray.util.state import list_objects


MIB = 1024 * 1024
RESULT_PREFIX = "ASCEND_MAZE_STAGE3A_RESULT="


@dataclass(frozen=True, slots=True)
class AdmissionConfig:
    expected_ray_version: str = "2.55.1"
    expected_cloudpickle_version: str = "3.1.2"
    object_store_bytes: int = 128 * MIB
    probe_payload_bytes: int = 8 * MIB
    churn_payload_bytes: int = 24 * MIB
    churn_objects_per_round: int = 7
    rounds: int = 5
    cleanup_deadline_seconds: float = 30.0
    worker_exit_deadline_seconds: float = 10.0
    poll_interval_seconds: float = 0.2
    object_count_tolerance: int = 24
    object_bytes_tolerance: int = 4 * MIB
    spill_bytes_tolerance: int = 0
    spill_tolerance_basis: str = "normal ray.put control returns to zero in one second"


@dataclass(frozen=True, slots=True)
class ObjectStoreSnapshot:
    object_count: int
    object_bytes: int
    objects: dict[str, int]
    spill_bytes: int

    def summary(self) -> dict[str, int]:
        return {
            "object_count": self.object_count,
            "object_bytes": self.object_bytes,
            "spill_bytes": self.spill_bytes,
        }


@ray.remote(num_cpus=0, max_restarts=0)
class DataStoreOwnerProbe:
    """Keep Ray-recognized ObjectRefs without materializing their payloads."""

    def __init__(self, owner_generation: str) -> None:
        self.owner_generation = owner_generation
        self._refs: dict[str, ray.ObjectRef] = {}
        self._object_ids: dict[str, str] = {}
        self._tombstones: set[str] = set()
        self.payload_materialization_count = 0

    def adopt(
        self,
        owner_generation: str,
        handle_id: str,
        boxed_ref: list[ray.ObjectRef],
    ) -> dict[str, str]:
        self._require_generation(owner_generation)
        if len(boxed_ref) != 1 or not isinstance(boxed_ref[0], ray.ObjectRef):
            raise TypeError("adopt requires one nested ObjectRef")
        object_ref = boxed_ref[0]
        object_id = object_ref.hex()
        existing = self._object_ids.get(handle_id)
        if existing is not None and existing != object_id:
            raise RuntimeError("handle_id already identifies another Ray object")
        self._refs[handle_id] = object_ref
        self._object_ids[handle_id] = object_id
        self._tombstones.discard(handle_id)
        return {
            "handle_id": handle_id,
            "object_id": object_id,
            "owner_generation": self.owner_generation,
        }

    def resolve(self, owner_generation: str, handle_id: str) -> ray.ObjectRef:
        self._require_generation(owner_generation)
        if handle_id in self._tombstones:
            raise RuntimeError("data handle is released")
        try:
            return self._refs[handle_id]
        except KeyError as exc:
            raise RuntimeError("data handle is unknown") from exc

    def release(self, owner_generation: str, handle_id: str) -> bool:
        self._require_generation(owner_generation)
        object_ref = self._refs.pop(handle_id, None)
        existed = object_ref is not None
        self._object_ids.pop(handle_id, None)
        self._tombstones.add(handle_id)
        del object_ref
        gc.collect()
        return existed

    def release_many(
        self,
        owner_generation: str,
        handle_ids: list[str],
    ) -> int:
        return sum(self.release(owner_generation, handle_id) for handle_id in handle_ids)

    def inspect_handle(self, owner_generation: str, handle_id: str) -> dict[str, Any]:
        if owner_generation != self.owner_generation:
            return {
                "readable": False,
                "state": "owner_generation_mismatch",
                "owner_generation": self.owner_generation,
            }
        if handle_id in self._tombstones:
            state = "released"
        elif handle_id in self._refs:
            state = "active"
        else:
            state = "unknown"
        return {
            "readable": state == "active",
            "state": state,
            "owner_generation": self.owner_generation,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "owner_generation": self.owner_generation,
            "active_ref_count": len(self._refs),
            "tombstone_count": len(self._tombstones),
            "payload_materialization_count": self.payload_materialization_count,
            "pid": os.getpid(),
        }

    def _require_generation(self, owner_generation: str) -> None:
        if owner_generation != self.owner_generation:
            raise RuntimeError("owner generation mismatch")


@ray.remote(num_cpus=1, max_calls=1)
def one_shot_worker_put(
    owner: ray.actor.ActorHandle,
    owner_generation: str,
    handle_id: str,
    payload_bytes: int,
    fill_byte: int,
) -> dict[str, Any]:
    payload = bytes((fill_byte,)) * payload_bytes
    object_ref = ray.put(payload, _owner=owner)
    adopted = ray.get(
        owner.adopt.remote(owner_generation, handle_id, [object_ref])
    )
    return {
        **adopted,
        "pid": os.getpid(),
        "payload_bytes": payload_bytes,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


@ray.remote(num_cpus=1)
def verify_without_controller_materialization(
    owner: ray.actor.ActorHandle,
    owner_generation: str,
    handle_id: str,
) -> dict[str, Any]:
    object_ref = ray.get(owner.resolve.remote(owner_generation, handle_id))
    payload = ray.get(object_ref)
    if not isinstance(payload, bytes):
        raise TypeError("probe payload must be bytes")
    return {
        "pid": os.getpid(),
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


@ray.remote(num_cpus=1)
def verify_invalid_handle(
    owner: ray.actor.ActorHandle,
    owner_generation: str,
    handle_id: str,
) -> dict[str, Any]:
    try:
        ray.get(owner.resolve.remote(owner_generation, handle_id))
    except Exception as exc:
        return {
            "rejected": True,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    return {"rejected": False}


def _payload_digest(payload_bytes: int, fill_byte: int) -> str:
    digest = hashlib.sha256()
    chunk = bytes((fill_byte,)) * min(payload_bytes, MIB)
    remaining = payload_bytes
    while remaining:
        part = chunk[:remaining]
        digest.update(part)
        remaining -= len(part)
    return digest.hexdigest()


def _spill_bytes(spill_directory: Path) -> int:
    if not spill_directory.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in spill_directory.rglob("*")
        if path.is_file()
    )


def _object_store_snapshot(spill_directory: Path) -> ObjectStoreSnapshot:
    records = list_objects(
        limit=10_000,
        timeout=30,
        detail=True,
        raise_on_missing_output=False,
    )
    objects: dict[str, int] = {}
    for record in records:
        size = max(0, int(record.object_size))
        objects[record.object_id] = max(objects.get(record.object_id, 0), size)
    return ObjectStoreSnapshot(
        object_count=len(objects),
        object_bytes=sum(objects.values()),
        objects=objects,
        spill_bytes=_spill_bytes(spill_directory),
    )


def _wait_for(
    description: str,
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except Exception as exc:
            last_error = exc
        time.sleep(poll_interval_seconds)
    suffix = "" if last_error is None else f"; last error: {last_error}"
    raise AssertionError(f"deadline waiting for {description}{suffix}")


def _process_exited(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _parse_child_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            value = json.loads(line.removeprefix(RESULT_PREFIX))
            if not isinstance(value, dict):
                break
            return value
    raise RuntimeError("RuntimeClient child did not emit a result record")


def _runtime_client_put(args: argparse.Namespace) -> int:
    ray.init(
        address=args.address,
        namespace=args.namespace,
        log_to_driver=False,
    )
    owner = ray.get_actor(args.owner_name, namespace=args.namespace)
    handles: list[dict[str, Any]] = []
    for index in range(args.count):
        fill_byte = (args.fill_byte + index) % 251 + 1
        payload = bytes((fill_byte,)) * args.payload_bytes
        object_ref = ray.put(payload, _owner=owner)
        handle_id = f"{args.handle_prefix}_{index}"
        adopted = ray.get(
            owner.adopt.remote(args.owner_generation, handle_id, [object_ref])
        )
        handles.append(
            {
                **adopted,
                "payload_bytes": args.payload_bytes,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "fill_byte": fill_byte,
            }
        )
    result = {"pid": os.getpid(), "handles": handles}
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    ray.shutdown()
    return 0


def _run_runtime_client(
    *,
    address: str,
    namespace: str,
    owner_name: str,
    owner_generation: str,
    handle_prefix: str,
    count: int,
    payload_bytes: int,
    fill_byte: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--client-put",
        "--address",
        address,
        "--namespace",
        namespace,
        "--owner-name",
        owner_name,
        "--owner-generation",
        owner_generation,
        "--handle-prefix",
        handle_prefix,
        "--count",
        str(count),
        "--payload-bytes",
        str(payload_bytes),
        "--fill-byte",
        str(fill_byte),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "RuntimeClient child failed: "
            + completed.stderr[-4000:]
            + completed.stdout[-4000:]
        )
    return _parse_child_result(completed.stdout)


def _api_audit(config: AdmissionConfig) -> dict[str, Any]:
    signature = inspect.signature(ray.put)
    owner_parameter = signature.parameters.get("_owner")
    annotation_type = getattr(ray.put, "_annotated_type", None)
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_tokens = ("ray." + "_private", "ray." + "internal")
    forbidden_hits = [token for token in forbidden_tokens if token in source]
    if ray.__version__ != config.expected_ray_version:
        raise AssertionError(
            f"expected Ray {config.expected_ray_version}, got {ray.__version__}"
        )
    if cloudpickle.__version__ != config.expected_cloudpickle_version:
        raise AssertionError(
            "expected cloudpickle "
            f"{config.expected_cloudpickle_version}, got {cloudpickle.__version__}"
        )
    if owner_parameter is None:
        raise AssertionError("target Ray has no ray.put owner argument")
    if str(annotation_type) != "AnnotationType.PUBLIC_API":
        raise AssertionError("ray.put is not annotated as a public API")
    if forbidden_hits:
        raise AssertionError(f"private Ray API references found: {forbidden_hits}")
    return {
        "ray_version": ray.__version__,
        "cloudpickle_version": cloudpickle.__version__,
        "ray_put_annotation": str(annotation_type),
        "owner_parameter": str(owner_parameter),
        "owner_parameter_stability": "experimental",
        "forbidden_private_api_hits": forbidden_hits,
    }


def _within_cleanup_tolerance(
    current: ObjectStoreSnapshot,
    baseline: ObjectStoreSnapshot,
    config: AdmissionConfig,
) -> bool:
    return (
        current.object_count <= baseline.object_count + config.object_count_tolerance
        and current.object_bytes
        <= baseline.object_bytes + config.object_bytes_tolerance
        and current.spill_bytes
        <= baseline.spill_bytes + config.spill_bytes_tolerance
    )


def _new_owner(
    *,
    owner_name: str,
    owner_generation: str,
) -> ray.actor.ActorHandle:
    return DataStoreOwnerProbe.options(
        name=owner_name,
        lifetime="detached",
        num_cpus=0,
    ).remote(owner_generation)


def _run_admission(config: AdmissionConfig, work_directory: Path) -> dict[str, Any]:
    api = _api_audit(config)
    spill_directory = work_directory / "spill"
    spill_directory.mkdir(parents=True, exist_ok=True)
    namespace = f"ascend-maze-stage3a-{uuid.uuid4().hex}"
    owner_name = f"data-store-owner-{uuid.uuid4().hex}"
    owner_generation = f"owner-{uuid.uuid4().hex}"
    context = ray.init(
        num_cpus=4,
        object_store_memory=config.object_store_bytes,
        object_spilling_directory=str(spill_directory),
        include_dashboard=True,
        dashboard_port=0,
        namespace=namespace,
        log_to_driver=False,
    )
    owner: ray.actor.ActorHandle | None = None
    new_generation_owner: ray.actor.ActorHandle | None = None
    rounds: list[dict[str, Any]] = []
    try:
        address = context.address_info["address"]
        owner = _new_owner(
            owner_name=owner_name,
            owner_generation=owner_generation,
        )
        ray.get(owner.stats.remote())
        _wait_for(
            "Ray State API",
            lambda: _object_store_snapshot(spill_directory).object_count >= 0,
            timeout_seconds=config.cleanup_deadline_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
        )
        gc.collect()
        baseline = _object_store_snapshot(spill_directory)
        peak_spill_bytes = baseline.spill_bytes
        peak_large_object_copies = 0

        for round_index in range(config.rounds):
            before = _object_store_snapshot(spill_directory)
            child = _run_runtime_client(
                address=address,
                namespace=namespace,
                owner_name=owner_name,
                owner_generation=owner_generation,
                handle_prefix=f"round_{round_index}_input",
                count=config.churn_objects_per_round,
                payload_bytes=config.churn_payload_bytes,
                fill_byte=round_index * 17,
            )
            client_pid = int(child["pid"])
            if not _process_exited(client_pid):
                raise AssertionError("RuntimeClient process did not exit")
            handles = list(child["handles"])
            handle_ids = [str(item["handle_id"]) for item in handles]
            object_ids = {str(item["object_id"]) for item in handles}

            active = _object_store_snapshot(spill_directory)
            missing = object_ids - set(active.objects)
            if missing:
                raise AssertionError(f"owner lost RuntimeClient objects: {sorted(missing)}")
            new_large_objects = {
                object_id
                for object_id, size in active.objects.items()
                if object_id not in before.objects
                and size >= int(config.churn_payload_bytes * 0.9)
            }
            unexpected_large = new_large_objects - object_ids
            peak_large_object_copies = max(
                peak_large_object_copies,
                len(unexpected_large),
            )
            if unexpected_large:
                raise AssertionError(
                    "ownership transfer created extra large Ray objects: "
                    f"{sorted(unexpected_large)}"
                )

            first = handles[0]
            verification = ray.get(
                verify_without_controller_materialization.remote(
                    owner,
                    owner_generation,
                    first["handle_id"],
                )
            )
            if verification["payload_sha256"] != first["payload_sha256"]:
                raise AssertionError("RuntimeClient object digest changed after exit")

            output_handle_id = f"round_{round_index}_worker_output"
            fill_byte = (round_index * 29) % 251 + 1
            worker_output = ray.get(
                one_shot_worker_put.remote(
                    owner,
                    owner_generation,
                    output_handle_id,
                    config.probe_payload_bytes,
                    fill_byte,
                )
            )
            worker_pid = int(worker_output["pid"])
            _wait_for(
                "one-shot Worker process exit",
                lambda: _process_exited(worker_pid),
                timeout_seconds=config.worker_exit_deadline_seconds,
                poll_interval_seconds=config.poll_interval_seconds,
            )
            worker_verification = ray.get(
                verify_without_controller_materialization.remote(
                    owner,
                    owner_generation,
                    output_handle_id,
                )
            )
            expected_worker_digest = _payload_digest(
                config.probe_payload_bytes,
                fill_byte,
            )
            if worker_verification["payload_sha256"] != expected_worker_digest:
                raise AssertionError("one-shot Worker output digest changed after exit")

            active_with_output = _object_store_snapshot(spill_directory)
            output_object_id = str(worker_output["object_id"])
            new_output_large_objects = {
                object_id
                for object_id, size in active_with_output.objects.items()
                if object_id not in active.objects
                and size >= int(config.probe_payload_bytes * 0.9)
            }
            unexpected_output_large = new_output_large_objects - {output_object_id}
            peak_large_object_copies = max(
                peak_large_object_copies,
                len(unexpected_output_large),
            )
            if output_object_id not in new_output_large_objects:
                raise AssertionError("one-shot Worker output is absent from Object Store")
            if unexpected_output_large:
                raise AssertionError(
                    "Worker ownership transfer created extra large Ray objects: "
                    f"{sorted(unexpected_output_large)}"
                )
            peak_spill_bytes = max(peak_spill_bytes, active_with_output.spill_bytes)
            all_handle_ids = [*handle_ids, output_handle_id]
            all_object_ids = {
                *object_ids,
                output_object_id,
            }
            released = ray.get(
                owner.release_many.remote(owner_generation, all_handle_ids)
            )
            if released != len(all_handle_ids):
                raise AssertionError("owner did not release every staged object")
            for handle_id in all_handle_ids:
                state = ray.get(
                    owner.inspect_handle.remote(owner_generation, handle_id)
                )
                if state != {
                    "readable": False,
                    "state": "released",
                    "owner_generation": owner_generation,
                }:
                    raise AssertionError("released handle did not become a tombstone")

            del handles, child, verification, worker_output, worker_verification
            gc.collect()

            last_cleanup_snapshot: ObjectStoreSnapshot | None = None

            def cleanup_complete() -> bool:
                nonlocal last_cleanup_snapshot
                snapshot = _object_store_snapshot(spill_directory)
                last_cleanup_snapshot = snapshot
                return not (all_object_ids & set(snapshot.objects)) and (
                    _within_cleanup_tolerance(snapshot, baseline, config)
                )

            cleanup_started = time.monotonic()
            try:
                _wait_for(
                    f"round {round_index} physical reclamation",
                    cleanup_complete,
                    timeout_seconds=config.cleanup_deadline_seconds,
                    poll_interval_seconds=config.poll_interval_seconds,
                )
            except AssertionError as exc:
                assert last_cleanup_snapshot is not None
                remaining_targets = sorted(
                    all_object_ids & set(last_cleanup_snapshot.objects)
                )
                raise AssertionError(
                    f"{exc}; baseline={baseline.summary()}; "
                    f"last={last_cleanup_snapshot.summary()}; "
                    f"remaining_target_ids={remaining_targets}"
                ) from exc
            after = _object_store_snapshot(spill_directory)
            cleanup_duration_seconds = time.monotonic() - cleanup_started
            rounds.append(
                {
                    "round": round_index,
                    "runtime_client_pid": client_pid,
                    "one_shot_worker_pid": worker_pid,
                    "active": active_with_output.summary(),
                    "reclaimed": after.summary(),
                    "cleanup_duration_seconds": cleanup_duration_seconds,
                    "target_objects_reclaimed": len(all_object_ids),
                }
            )

        if peak_spill_bytes <= baseline.spill_bytes:
            raise AssertionError("churn did not exercise Ray object spilling")
        if peak_large_object_copies != 0:
            raise AssertionError("large-object copy audit failed")

        generation_probe = _run_runtime_client(
            address=address,
            namespace=namespace,
            owner_name=owner_name,
            owner_generation=owner_generation,
            handle_prefix="generation_probe",
            count=1,
            payload_bytes=config.probe_payload_bytes,
            fill_byte=241,
        )
        old_handle = generation_probe["handles"][0]
        old_object_id = str(old_handle["object_id"])
        old_owner_stats = ray.get(owner.stats.remote())
        old_owner_pid = int(old_owner_stats["pid"])
        ray.kill(owner, no_restart=True)
        owner = None
        _wait_for(
            "old DataStoreOwner process exit",
            lambda: _process_exited(old_owner_pid),
            timeout_seconds=config.worker_exit_deadline_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
        )
        new_generation = f"owner-{uuid.uuid4().hex}"

        def create_replacement() -> bool:
            nonlocal new_generation_owner
            try:
                new_generation_owner = _new_owner(
                    owner_name=owner_name,
                    owner_generation=new_generation,
                )
                ray.get(new_generation_owner.stats.remote())
                return True
            except Exception:
                new_generation_owner = None
                return False

        _wait_for(
            "replacement DataStoreOwner generation",
            create_replacement,
            timeout_seconds=config.cleanup_deadline_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
        )
        assert new_generation_owner is not None
        generation_state = ray.get(
            new_generation_owner.inspect_handle.remote(
                owner_generation,
                old_handle["handle_id"],
            )
        )
        if generation_state["state"] != "owner_generation_mismatch":
            raise AssertionError("old owner generation was not rejected")
        invalid_read = ray.get(
            verify_invalid_handle.remote(
                new_generation_owner,
                owner_generation,
                old_handle["handle_id"],
            )
        )
        if not invalid_read["rejected"]:
            raise AssertionError("old owner generation remained readable")

        del generation_probe, old_handle
        gc.collect()
        _wait_for(
            "old owner generation object reclamation",
            lambda: old_object_id
            not in _object_store_snapshot(spill_directory).objects,
            timeout_seconds=config.cleanup_deadline_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
        )
        final_snapshot = _object_store_snapshot(spill_directory)
        if not _within_cleanup_tolerance(final_snapshot, baseline, config):
            raise AssertionError("final Ray storage usage did not return to tolerance")
        new_owner_stats = ray.get(new_generation_owner.stats.remote())
        if new_owner_stats["payload_materialization_count"] != 0:
            raise AssertionError("DataStoreOwner materialized a payload")

        return {
            "api": api,
            "ray_address": address,
            "namespace": namespace,
            "metrics": {
                "object_store": "ray.util.state.list_objects(detail=True)",
                "spill": "configured spill directory recursive file bytes",
                "worker_exit": "same-node process liveness via os.kill(pid, 0)",
            },
            "baseline": baseline.summary(),
            "final": final_snapshot.summary(),
            "peak_spill_bytes": peak_spill_bytes,
            "unexpected_large_copy_count": peak_large_object_copies,
            "controller_payload_materialization_count": 0,
            "owner_payload_materialization_count": new_owner_stats[
                "payload_materialization_count"
            ],
            "runtime_client_exit_verified": True,
            "one_shot_worker_exit_verified": True,
            "logical_destroy_invalidation_verified": True,
            "generation_invalidation_verified": True,
            "rounds": rounds,
        }
    finally:
        for actor in (owner, new_generation_owner):
            if actor is not None:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    pass
        ray.shutdown()


def _write_report(report_path: Path, report: dict[str, Any]) -> str:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/stage3a/ray_owner_admission.json"),
    )
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument(
        "--expected-ray-version",
        default=AdmissionConfig().expected_ray_version,
    )
    parser.add_argument("--client-put", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--address", help=argparse.SUPPRESS)
    parser.add_argument("--namespace", help=argparse.SUPPRESS)
    parser.add_argument("--owner-name", help=argparse.SUPPRESS)
    parser.add_argument("--owner-generation", help=argparse.SUPPRESS)
    parser.add_argument("--handle-prefix", help=argparse.SUPPRESS)
    parser.add_argument("--count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--payload-bytes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--fill-byte", type=int, default=1, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.client_put:
        required = (
            args.address,
            args.namespace,
            args.owner_name,
            args.owner_generation,
            args.handle_prefix,
            args.payload_bytes,
        )
        if any(value is None for value in required):
            raise SystemExit("client-put mode requires Ray and owner arguments")
        return _runtime_client_put(args)

    config = AdmissionConfig(expected_ray_version=args.expected_ray_version)
    work_directory = args.work_directory or Path(
        f"/tmp/ascend-maze-stage3a-{uuid.uuid4().hex}"
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
        "config": asdict(config),
        "work_directory": str(work_directory),
        "decision": "failed",
    }
    exit_code = 1
    try:
        report["evidence"] = _run_admission(config, work_directory)
        report["decision"] = "passed"
        exit_code = 0
    except Exception as exc:
        report["failure"] = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report_sha256 = _write_report(args.report, report)
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "report": str(args.report),
                    "report_sha256": report_sha256,
                },
                sort_keys=True,
            )
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
