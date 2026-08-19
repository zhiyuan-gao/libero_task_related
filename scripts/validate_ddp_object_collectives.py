#!/usr/bin/env python3
"""Smoke-test NCCL tensor collectives alongside CPU checkpoint object gathering."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    object_group = dist.new_group(backend="gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(
            {"rank": rank, "cpu_rng_bytes": int(torch.get_rng_state().numel())},
            gathered,
            dst=0,
            group=object_group,
        )
        if rank == 0:
            assert gathered is not None
            assert [item["rank"] for item in gathered] == list(range(world_size))

        value = torch.tensor(float(rank), device=device)
        dist.all_reduce(value)
        assert value.item() == world_size * (world_size - 1) / 2
        dist.barrier()
        if rank == 0:
            print(f"PASS: gathered {world_size} CPU objects and preserved NCCL all-reduce")
    finally:
        dist.destroy_process_group(object_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
