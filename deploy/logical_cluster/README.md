# Eight-node logical Ascend cluster

This deployment divides the physical 8-NPU host into eight Docker compute
nodes. Each node receives one logical NPU, 20 exclusive CPU cores, a 240 GiB
memory limit and a 16 GiB `/dev/shm`.

The deployment does not copy CANN, Conda environments, source code or model
weights. It mounts the host installations read-only and stores mutable state
under:

```text
~/.local/state/ascend-maze/logical-cluster/
```

## Commands

```bash
cd /home/user2/workplace/Ascend-Maze

deploy/logical_cluster/logical_cluster.sh up
deploy/logical_cluster/logical_cluster.sh verify
deploy/logical_cluster/logical_cluster.sh verify-binding 3
deploy/logical_cluster/logical_cluster.sh status
deploy/logical_cluster/logical_cluster.sh control-up
deploy/logical_cluster/logical_cluster.sh control-status
deploy/logical_cluster/logical_cluster.sh shell 0
deploy/logical_cluster/logical_cluster.sh exec 0 python -V
deploy/logical_cluster/logical_cluster.sh control-down
deploy/logical_cluster/logical_cluster.sh down
```

`verify` checks CPU affinity, the cgroup memory limit, single-device visibility
and a real NPU Tensor operation on every node. It also verifies the Ray and
Ascend-Maze imports and both shared model directories.

`verify-binding N` constructs a real `DeviceBinding` in a child Worker. It
checks that CANN uses runtime device `0`, DCMI reports the Worker only on host
physical NPU `N`, and HBM returns to the pre-Worker baseline after exit.

`control-up` uses node-0 as the Ray Head, Controller and first NodeAgent. It
joins node-1 through node-7 as Ray workers and NodeAgents, then waits until both
the Controller and Ray report eight healthy logical nodes. Private generated
configuration and the cluster token remain under the state root; they are not
written to the repository.

## End-to-end acceptance

Run the cold text and vision acceptance sequence after `control-up`:

```bash
deploy/logical_cluster/logical_cluster.sh exec 0 \
  /home/user2/workplace/miniconda3/envs/ascend-maze/bin/python \
  /home/user2/workplace/Ascend-Maze/tools/logical_cluster_e2e.py \
  --family all \
  --output-dir /workspace/state/output/logical-cluster-e2e-all
```

The cold text Run may place every Task on the same node because its model
reservation does not exist when the first Task is placed. After the first
command has established the logical model instances, require direct
cross-node evidence with:

```bash
deploy/logical_cluster/logical_cluster.sh exec 0 \
  /home/user2/workplace/miniconda3/envs/ascend-maze/bin/python \
  /home/user2/workplace/Ascend-Maze/tools/logical_cluster_e2e.py \
  --family text \
  --require-cross-node-text \
  --output-dir /workspace/state/output/logical-cluster-e2e-text-crossnode
```

Each Run materializes its exit result, destroys its Run data index and waits
for Run-owned leases, active Worker leases and used-device HBM to recover.
`text.json`, `vision.json` and `summary.json` retain the task nodes, physical
NPU evidence, timing, destroy tombstone and recovery snapshots.

The default openEuler image is pinned to the tested ARM64 multi-architecture
manifest digest. Set `ASCEND_MAZE_CONTAINER_IMAGE` only when intentionally
testing another compatible image.

## Topology

| Node | Physical NPU | CPUs | NUMA memory nodes | Address |
|---|---:|---|---|---|
| node-0 | 0 | 144-153,168-177 | 6,7 | 172.30.240.10 |
| node-1 | 1 | 156-165,180-189 | 6,7 | 172.30.240.11 |
| node-2 | 2 | 96-105,120-129 | 4,5 | 172.30.240.12 |
| node-3 | 3 | 108-117,132-141 | 4,5 | 172.30.240.13 |
| node-4 | 4 | 0-9,24-33 | 0,1 | 172.30.240.14 |
| node-5 | 5 | 12-21,36-45 | 0,1 | 172.30.240.15 |
| node-6 | 6 | 48-57,72-81 | 2,3 | 172.30.240.16 |
| node-7 | 7 | 60-69,84-93 | 2,3 | 172.30.240.17 |

The host retains 32 CPUs for the Controller, benchmark client and independent
resource monitor. Two containers share each pair of NUMA memory nodes because
the physical topology attaches two NPUs to one local CPU NUMA node.

This is one physical host with eight logical nodes. It is suitable for resource
placement and colocation experiments, but it does not reproduce physical
multi-node network bandwidth, latency or failure isolation.

## Runtime integration boundary

Docker maps host `/dev/davinciN` to `/dev/davinci0` in node N. CANN execution
therefore uses logical device 0, while DCMI and `npu-smi` retain the host
physical device identity N. Each generated NodeAgent configuration explicitly
declares:

```text
physical_device_id=N
runtime_visible_device_id=0
visible_device_index=0
```

The NodeAgent sends this topology in its registration. The Controller stores
it in `RuntimeNodeBinding`, and `DeviceBinding` uses it to configure CANN while
retaining the physical ID for DCMI verification. On bare metal, an omitted
mapping defaults to the compatible identity mapping `N -> N -> 0`.
