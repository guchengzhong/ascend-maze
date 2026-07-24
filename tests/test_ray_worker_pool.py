from __future__ import annotations

import asyncio

import pytest

import ascend_maze.runtime.ray_worker_pool as ray_worker_pool


class _BlockingProcessProbe:
    def options(self, **kwargs: object) -> "_BlockingProcessProbe":
        del kwargs
        return self

    def remote(self, process_id: int) -> str:
        assert process_id == 123
        return "process-probe-ref"


def test_worker_termination_bounds_a_stuck_process_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[object] = []
    cancelled: list[tuple[object, bool]] = []

    def blocked_get(ref: object) -> object:
        del ref
        raise AssertionError("blocked Ray get must run through the async stub")

    async def controlled_to_thread(
        function: object, *args: object, **kwargs: object
    ) -> object:
        del kwargs
        if function is blocked_get:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        assert callable(function)
        return function(*args)

    monkeypatch.setattr(ray_worker_pool, "_RAY_PROCESS_PROBE", _BlockingProcessProbe())
    monkeypatch.setattr(ray_worker_pool.asyncio, "to_thread", controlled_to_thread)
    monkeypatch.setattr(ray_worker_pool.ray, "get", blocked_get)
    monkeypatch.setattr(
        ray_worker_pool.ray,
        "kill",
        lambda actor, *, no_restart: killed.append((actor, no_restart)),
    )
    monkeypatch.setattr(
        ray_worker_pool.ray,
        "cancel",
        lambda ref, *, force: cancelled.append((ref, force)),
    )
    actor = object()
    endpoint = ray_worker_pool._RayWorkerEndpoint(  # noqa: SLF001
        actor=actor,
        ray_node_id="1" * 56,
        worker_pid=123,
    )

    with pytest.raises(TimeoutError, match="PID 123"):
        asyncio.run(
            ray_worker_pool.RayWorkerEndpointFactory().terminate(
                endpoint,
                force=True,
                timeout_ms=10,
            )
        )

    assert cancelled == [("process-probe-ref", True)]
    assert killed == [(actor, True), (actor, True)]
