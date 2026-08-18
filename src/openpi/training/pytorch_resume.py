"""Trajectory-exact PyTorch trainer state capture and restoration."""

from __future__ import annotations

import random

import numpy as np
import torch


def capture_training_state(data_loader, *, micro_step_in_update: int, rank: int, world_size: int) -> dict:
    """Capture per-rank RNG and data-order state at an optimizer boundary."""

    if micro_step_in_update != 0:
        raise ValueError("Exact checkpoints may only be saved at a gradient-accumulation boundary")
    if rank < 0 or world_size < 1 or rank >= world_size:
        raise ValueError(f"Invalid distributed identity: rank={rank}, world_size={world_size}")
    if not hasattr(data_loader, "state_dict"):
        raise TypeError("Data loader does not expose exact resume state")
    cuda_rng_state = None
    if torch.cuda.is_available():
        cuda_device = torch.cuda.current_device()
        cuda_rng_state = {
            "device": cuda_device,
            "state": torch.cuda.get_rng_state(cuda_device),
        }
    return {
        "schema": "openpi.pytorch_rank_resume_state.v1",
        "rank": rank,
        "world_size": world_size,
        "micro_step_in_update": micro_step_in_update,
        "torch_cpu_rng_state": torch.get_rng_state(),
        # DDP owns one CUDA device per process, so each gathered rank payload
        # records only its local generator instead of touching every GPU.
        "torch_cuda_rng_state": cuda_rng_state,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "data_loader_state": data_loader.state_dict(),
    }


def restore_training_state(state: dict, data_loader, *, rank: int, world_size: int) -> None:
    """Restore one rank after model and optimizer checkpoint loading is complete."""

    if state.get("schema") != "openpi.pytorch_rank_resume_state.v1":
        raise ValueError(f"Unsupported rank resume state: {state.get('schema')}")
    if int(state["rank"]) != rank or int(state["world_size"]) != world_size:
        raise ValueError(
            "Exact resume requires the original distributed topology: "
            f"saved rank/world={state['rank']}/{state['world_size']}, current={rank}/{world_size}"
        )
    if int(state["micro_step_in_update"]) != 0:
        raise ValueError("Saved checkpoint is not at a gradient-accumulation boundary")
    if not hasattr(data_loader, "load_state_dict"):
        raise TypeError("Data loader does not support exact resume state")

    data_loader.load_state_dict(state["data_loader_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    saved_cuda_state = state["torch_cuda_rng_state"]
    if torch.cuda.is_available():
        if saved_cuda_state is None:
            raise ValueError("Checkpoint lacks the CUDA RNG state required by the current CUDA run")
        current_device = torch.cuda.current_device()
        if int(saved_cuda_state["device"]) != current_device:
            raise ValueError(
                "Exact CUDA RNG resume requires the original local device mapping: "
                f"saved={saved_cuda_state['device']}, current={current_device}"
            )
        torch.cuda.set_rng_state(saved_cuda_state["state"].cpu(), device=current_device)
    elif saved_cuda_state is not None:
        raise ValueError("Checkpoint contains CUDA RNG states but CUDA is unavailable")
    np.random.set_state(state["numpy_rng_state"])
    random.setstate(state["python_rng_state"])
