from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
import ray
from ray.cluster_utils import Cluster


@pytest.fixture(scope="session")
def ray_namespace() -> Iterator[str]:
    namespace = f"ascend-maze-test-{uuid4().hex}"
    cluster = Cluster(shutdown_at_exit=False)
    cluster.add_node(
        num_cpus=2,
        object_store_memory=128 * 1024 * 1024,
        include_dashboard=False,
    )
    cluster.add_node(
        num_cpus=2,
        object_store_memory=128 * 1024 * 1024,
    )
    ray.init(address=cluster.address, namespace=namespace, log_to_driver=False)
    try:
        yield namespace
    finally:
        ray.shutdown()
        cluster.shutdown()


@pytest.fixture(scope="session")
def ray_node_ids(ray_namespace: str) -> tuple[str, str]:
    del ray_namespace
    node_ids = tuple(
        sorted(
            str(node["NodeID"])
            for node in ray.nodes()
            if bool(node.get("Alive"))
        )
    )
    if len(node_ids) != 2:
        raise RuntimeError(f"expected two live Ray nodes, got {node_ids}")
    return node_ids
