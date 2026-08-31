"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import contextlib
import dataclasses
import gc
import logging
import os
import platform
import random
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi05_aux_queries as _pi05_aux
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.training.pytorch_ema as _pytorch_ema
import openpi.training.pytorch_resume as _pytorch_resume

_OBJECT_PROCESS_GROUP: dist.ProcessGroup | None = None


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    global _OBJECT_PROCESS_GROUP  # noqa: PLW0603

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            device_id=device if backend == "nccl" else None,
        )
        # Object collectives pickle CPU-side RNG/data-loader state. PyTorch's
        # NCCL object path materializes its internal tensors on cuda:0, which
        # conflicts with rank-local device binding. Keep model collectives on
        # NCCL and serialize checkpoint metadata over an explicit CPU group.
        _OBJECT_PROCESS_GROUP = dist.new_group(backend="gloo")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    return use_ddp, local_rank, device


def cleanup_ddp():
    global _OBJECT_PROCESS_GROUP  # noqa: PLW0603

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        if _OBJECT_PROCESS_GROUP is not None:
            torch.distributed.destroy_process_group(_OBJECT_PROCESS_GROUP)
            _OBJECT_PROCESS_GROUP = None
        torch.distributed.destroy_process_group()


def set_seed(seed: int, rank: int):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def policy_aux_targets_from_batch(batch: dict) -> _pi05_aux.PolicyAuxTargets:
    """Convert the collated immutable target tree into the model target dataclass."""

    return _pi05_aux.PolicyAuxTargets(
        geometry=batch["geometry"],
        geometry_valid=batch["geometry_valid"],
        geometry_mean=batch["geometry_mean"],
        geometry_std=batch["geometry_std"],
        motion=batch.get("motion"),
        motion_valid=batch.get("motion_valid"),
        motion_mean=batch.get("motion_mean"),
        motion_std=batch.get("motion_std"),
        ground_masks=batch.get("ground_masks"),
        ground_valid_views=batch.get("ground_valid_views"),
        semantic_input_ids=batch.get("semantic_input_ids"),
        semantic_labels=batch.get("semantic_labels"),
        semantic_loss_mask=batch.get("semantic_loss_mask"),
    )


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def reduce_scalar_metrics(values: dict[str, float], device: torch.device) -> dict[str, float]:
    """Average scalar training metrics over DDP ranks in a fixed key order."""

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return values
    keys = sorted(values)
    packed = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed /= dist.get_world_size()
    return dict(zip(keys, packed.cpu().tolist(), strict=True))


def should_save_checkpoint(global_step: int, config: _config.TrainConfig) -> bool:
    interval = config.save_interval
    if config.late_save_start_step is not None and global_step > config.late_save_start_step:
        interval = config.late_save_interval
    return (global_step % interval == 0 and global_step > 0) or (
        config.save_final_checkpoint and global_step == config.num_train_steps
    )


_RESUME_RUNTIME_FIELDS = {
    "checkpoint_keep_steps",
    "keep_period",
    "late_save_interval",
    "late_save_start_step",
    "log_interval",
    "max_checkpoints_to_keep",
    "max_resume_checkpoints_to_keep",
    "num_train_steps",
    "overwrite",
    "resume",
    "save_final_checkpoint",
    "save_interval",
    "wandb_enabled",
}


def trajectory_config(config: _config.TrainConfig | dict) -> dict:
    """Return only fields that can affect the optimizer/data trajectory."""

    payload = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)
    return {key: value for key, value in payload.items() if key not in _RESUME_RUNTIME_FIELDS}


def prune_checkpoints(
    checkpoint_dir,
    *,
    keep_period: int | None,
    keep_steps: tuple[int, ...] = (),
    max_to_keep: int | None = None,
) -> list[int]:
    """Prune checkpoints while honoring periodic, exact, and rolling retention."""

    checkpoints = sorted(
        (int(path.name), path) for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )
    if not checkpoints:
        return []
    latest_step = checkpoints[-1][0]
    explicit = set(keep_steps)
    rolling = {step for step, _ in checkpoints[-max_to_keep:]} if max_to_keep is not None else {latest_step}
    removed = []
    for step, path in checkpoints:
        protected = (keep_period is not None and step % keep_period == 0) or step in explicit
        if step not in rolling and not protected:
            shutil.rmtree(path)
            removed.append(step)
    return removed


_EXACT_RESUME_FILES = ("optimizer.pt", "training_state.pt")


def checkpoint_is_resumable(path) -> bool:
    """Return whether a published checkpoint contains exact-continuation state."""

    required = (*_EXACT_RESUME_FILES, "metadata.pt")
    has_training_model = (path / "train_model.safetensors").is_file()
    has_standard_model = (path / "model.safetensors").is_file()
    return all((path / name).is_file() for name in required) and (has_training_model or has_standard_model)


def demote_old_resume_checkpoints(checkpoint_dir, *, max_to_keep: int | None) -> list[int]:
    """Keep exact continuation only for the newest checkpoints.

    Demotion happens only after a new checkpoint has been atomically published.
    Evaluation weights, assets, and metadata remain in every checkpoint.
    """

    if max_to_keep is None:
        return []
    resumable = sorted(
        (int(path.name), path)
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and checkpoint_is_resumable(path)
    )
    demoted = []
    for step, path in resumable[:-max_to_keep]:
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"Refusing to demote checkpoint without evaluation weights: {path}")
        for name in (*_EXACT_RESUME_FILES, "train_model.safetensors"):
            payload = path / name
            if payload.exists():
                payload.unlink()
        (path / "EVALUATION_ONLY").write_text(
            "Exact-continuation state was pruned; model.safetensors remains valid for evaluation.\n"
        )
        demoted.append(step)
    return demoted


def save_checkpoint(
    model,
    optimizer,
    global_step,
    config,
    is_main,
    data_config,
    data_loader,
    ema,
    *,
    micro_step_in_update: int,
):
    """Save model/optimizer plus per-rank exact-continuation state."""
    if not should_save_checkpoint(global_step, config):
        return

    model_for_device = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    parameter_device = next(model_for_device.parameters()).device
    if parameter_device.type == "cuda" and torch.cuda.current_device() != parameter_device.index:
        raise RuntimeError(
            "Refusing to save non-exact CUDA RNG state: "
            f"model device={parameter_device.index}, current device={torch.cuda.current_device()}"
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_training_state = _pytorch_resume.capture_training_state(
        data_loader,
        micro_step_in_update=micro_step_in_update,
        rank=rank,
        world_size=world_size,
    )
    if dist.is_initialized():
        if _OBJECT_PROCESS_GROUP is None:
            raise RuntimeError("DDP checkpointing requires the CPU object process group")
        rank_states = [None] * world_size if is_main else None
        dist.gather_object(local_training_state, rank_states, dst=0, group=_OBJECT_PROCESS_GROUP)
    else:
        rank_states = [local_training_state]
    if not is_main:
        return

    if rank_states is None or any(state is None for state in rank_states):
        raise RuntimeError("Failed to gather exact-resume state from every DDP rank")

    # Only rank 0 writes the atomically published checkpoint.
    final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
    tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

    # Remove any existing temp directory and create new one
    if tmp_ckpt_dir.exists():
        shutil.rmtree(tmp_ckpt_dir)
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save raw optimizer-updated weights for exact continuation and EMA weights
    # under the standard serving filename.
    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if ema is None:
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")
    else:
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "train_model.safetensors")
        ema.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

    # Save optimizer state using PyTorch format
    torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

    # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
    metadata = {
        "global_step": global_step,
        "config": dataclasses.asdict(config),
        "timestamp": time.time(),
        "resume_semantics": "EXACT_CONTINUATION",
        "gradient_accumulation_boundary": micro_step_in_update,
        "checkpoint_schema": "openpi.pytorch_raw_ema_checkpoint.v2",
        "ema": None if ema is None else ema.metadata(),
    }
    torch.save(metadata, tmp_ckpt_dir / "metadata.pt")
    torch.save(
        {
            "schema": "openpi.pytorch_resume_state.v1",
            "world_size": world_size,
            "rank_states": rank_states,
        },
        tmp_ckpt_dir / "training_state.pt",
    )

    # save norm stats
    norm_stats = data_config.norm_stats
    if norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

    # Atomically publish only after every required file is complete.
    if final_ckpt_dir.exists():
        raise FileExistsError(f"Refusing to replace an already-published checkpoint: {final_ckpt_dir}")
    tmp_ckpt_dir.rename(final_ckpt_dir)

    removed = prune_checkpoints(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        keep_steps=config.checkpoint_keep_steps,
        max_to_keep=config.max_checkpoints_to_keep,
    )
    if removed:
        logging.info(f"Pruned superseded ordinary checkpoints: {removed}")

    demoted = demote_old_resume_checkpoints(
        config.checkpoint_dir,
        max_to_keep=config.max_resume_checkpoints_to_keep,
    )
    if demoted:
        logging.info(f"Demoted old checkpoints to evaluation-only: {demoted}")

    logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

    # Log checkpoint to wandb
    if config.wandb_enabled:
        wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device, data_loader, config, ema):
    """Load the latest checkpoint and restore exact per-rank continuation state."""
    checkpoint_steps = [
        int(d.name) for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit() and checkpoint_is_resumable(d)
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No resumable checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    metadata_path = ckpt_dir / "metadata.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata is missing: {metadata_path}")
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    global_step = int(metadata.get("global_step", latest_step))
    if global_step != latest_step:
        raise RuntimeError(
            f"Checkpoint directory/metadata step mismatch: directory={latest_step}, metadata={global_step}"
        )
    if metadata.get("resume_semantics") != "EXACT_CONTINUATION":
        raise RuntimeError(
            "Checkpoint predates exact-continuation state. Resume is intentionally refused instead of "
            "silently restarting the data stream."
        )
    saved_config = metadata.get("config")
    if saved_config is None:
        raise RuntimeError("Exact-resume checkpoint is missing its training config")
    if trajectory_config(saved_config) != trajectory_config(config):
        raise RuntimeError("Exact resume requires the original trajectory-affecting training config")
    if config.num_train_steps < global_step:
        raise RuntimeError(
            f"Resume target num_train_steps={config.num_train_steps} is behind checkpoint step {global_step}"
        )
    saved_ema_metadata = metadata.get("ema")
    if ema is None and saved_ema_metadata is not None:
        raise RuntimeError("Checkpoint contains EMA state but the current config disables EMA")
    if ema is not None:
        if saved_ema_metadata is None:
            raise RuntimeError("EMA-enabled exact resume requires saved EMA metadata")
        if int(saved_ema_metadata.get("num_updates", -1)) != global_step:
            raise RuntimeError(
                f"EMA update count must equal optimizer step: ema={saved_ema_metadata.get('num_updates')}, "
                f"step={global_step}"
            )

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / ("train_model.safetensors" if ema is not None else "model.safetensors")

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, strict=True, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            # Deserialize large optimizer tensors into host memory first.
            # optimizer.load_state_dict then casts/moves each state tensor to
            # its parameter's rank-local device without stressing CUDA's
            # serialization path from four concurrent readers.
            optimizer_state_dict = torch.load(optimizer_path, map_location="cpu", weights_only=False)
            logging.info("Loaded optimizer state from pt format into host memory")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        if ema is not None:
            logging.info("Loading EMA inference parameters...")
            ema_path = ckpt_dir / "model.safetensors"
            if not ema_path.exists():
                raise FileNotFoundError(f"EMA inference parameters are missing: {ema_path}")
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            # Validate schema/dtypes before reading a multi-GB EMA file. In
            # particular, v1 BF16 shadows must not be silently upcast and
            # mistaken for full-precision EMA state.
            ema.load_metadata(saved_ema_metadata, model_to_load)
            ema.load_model(model_to_load, ema_path, device=str(device))
            logging.info(f"Restored EMA parameters at update {ema.num_updates}")

        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        training_state_path = ckpt_dir / "training_state.pt"
        if not training_state_path.exists():
            raise FileNotFoundError(f"Exact-resume state is missing: {training_state_path}")
        training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        if training_state.get("schema") != "openpi.pytorch_resume_state.v1":
            raise RuntimeError(f"Unsupported exact-resume payload: {training_state.get('schema')}")
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if int(training_state["world_size"]) != world_size:
            raise RuntimeError(
                "Exact resume requires the original world size: "
                f"saved={training_state['world_size']}, current={world_size}"
            )
        rank_states = training_state["rank_states"]
        if len(rank_states) != world_size:
            raise RuntimeError("Exact-resume payload does not contain one state per rank")
        _pytorch_resume.restore_training_state(
            rank_states[rank],
            data_loader,
            rank=rank,
            world_size=world_size,
        )

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir, *, resumable_only: bool = False):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and (not resumable_only or checkpoint_is_resumable(d))
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if os.environ.get("OPENPI_LOG_MEMORY_STATS", "1").lower() in {"0", "false", "no"}:
        return
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    if config.policy_aux is not None and not config.policy_aux.loss_coefficients_approved:
        raise RuntimeError(
            "P1/P2 loss coefficients are not approved. Run calibration and explicitly set "
            "policy_aux.loss_coefficients_approved=true before any optimizer step."
        )
    use_default_cuda_allocator = os.environ.get("OPENPI_USE_DEFAULT_CUDA_ALLOCATOR", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if use_default_cuda_allocator:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    elif int(os.environ.get("WORLD_SIZE", "1")) >= 8:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    rank = dist.get_rank() if use_ddp else 0
    set_seed(config.seed, rank)

    # Initialize checkpoint directory and wandb
    # Every rank must snapshot the pre-launch state before rank 0 can create the
    # directory. Otherwise slower ranks can mistake rank 0's new directory for
    # a pre-existing experiment and abort a clean DDP launch.
    checkpoint_dir_existed_at_launch = config.checkpoint_dir.exists()
    if use_ddp:
        dist.barrier()
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if checkpoint_dir_existed_at_launch:
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir, resumable_only=True)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite:
        if is_main and checkpoint_dir_existed_at_launch:
            shutil.rmtree(config.checkpoint_dir)
            logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")
        if use_ddp:
            dist.barrier()
    elif checkpoint_dir_existed_at_launch:
        raise FileExistsError(
            f"Checkpoint directory {config.checkpoint_dir} already exists. Use --overwrite or --resume."
        )

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        if is_main:
            exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
        if use_ddp:
            dist.barrier()
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    if config.batch_size % world_size != 0:
        raise ValueError("Global micro-batch size must be divisible by DDP world size")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    micro_batch_per_gpu = config.batch_size // world_size
    effective_global_batch = config.batch_size * config.gradient_accumulation_steps
    logging.info(
        f"Using micro-batch per GPU: {micro_batch_per_gpu}; global micro-batch: "
        f"{config.batch_size}; accumulation: {config.gradient_accumulation_steps}; "
        f"effective global batch: {effective_global_batch}"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch[:2]
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = _pi05_aux.create_pytorch_model(config, model_config=model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        if isinstance(model_to_load, _pi05_aux.PI05AuxPolicy) and model_to_load.aux_enabled:
            load_result = model_to_load.load_official_base_checkpoint(model_path, device=str(device))
            logging.info(f"Loaded official base with exact auxiliary missing keys: {load_result}")
        else:
            safetensors.torch.load_model(model_to_load, model_path, strict=True)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    model_for_state = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    ema = (
        None
        if config.ema_decay is None
        else _pytorch_ema.ExponentialMovingAverage(model_for_state, decay=config.ema_decay)
    )
    if ema is not None:
        logging.info(
            f"Initialized EMA after base checkpoint load: decay={ema.decay}, parameters={len(ema.metadata()['parameter_names'])}"
        )

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device, loader, config, ema)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    optim.zero_grad(set_to_none=True)
    micro_step_in_update = 0
    accumulated_loss = 0.0
    accumulated_auxiliary_values: dict[str, float] = {}
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: micro_batch_per_gpu={micro_batch_per_gpu}, "
            f"effective_global_batch={effective_global_batch}, "
            f"num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA: disabled" if ema is None else f"EMA: decay={ema.decay}, update once per optimizer update")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for batch in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            if config.policy_aux is None:
                observation, actions = batch
                policy_aux_targets = None
            else:
                observation, actions, policy_aux_batch = batch
                policy_aux_batch = jax.tree.map(lambda x: x.to(device), policy_aux_batch)
                policy_aux_targets = policy_aux_targets_from_batch(policy_aux_batch)

            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32)
            actions = actions.to(device)

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            is_update_boundary = micro_step_in_update + 1 == config.gradient_accumulation_steps
            sync_context = (
                contextlib.nullcontext()
                if not isinstance(model, torch.nn.parallel.DistributedDataParallel) or is_update_boundary
                else model.no_sync()
            )
            with sync_context:
                # Forward pass. DDP synchronization is deferred until the last
                # micro-batch of an optimizer update.
                if policy_aux_targets is None:
                    losses = model(observation, actions)
                else:
                    losses = model(observation, actions, aux_targets=policy_aux_targets)
                auxiliary_log_values = {}
                if isinstance(losses, dict):
                    auxiliary_log_values = {
                        f"loss_{name}": float(value.detach()) for name, value in losses["losses"].items()
                    }
                    auxiliary_log_values.update(
                        {
                            name: float(value.detach())
                            for name, value in losses["diagnostics"].items()
                            if value.ndim == 0
                        }
                    )
                    losses = losses["losses"]["total"]
                # Ensure losses is a tensor and handle different return types.
                if isinstance(losses, list | tuple):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=device, dtype=torch.float32)

                loss = losses.mean()
                (loss / config.gradient_accumulation_steps).backward()

            accumulated_loss += loss.item() / config.gradient_accumulation_steps
            for key, value in auxiliary_log_values.items():
                accumulated_auxiliary_values[key] = (
                    accumulated_auxiliary_values.get(key, 0.0) + value / config.gradient_accumulation_steps
                )
            micro_step_in_update += 1
            if not is_update_boundary:
                continue

            # Log memory usage after the synchronized backward pass.
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Match the official trainer: clip once and use the returned pre-clipping
            # global norm for monitoring. Per-parameter auxiliary norm collection
            # causes a device synchronization for every parameter and does not affect
            # optimization.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.optimizer.clip_gradient_norm,
                foreach=True,
            )

            # Optimizer step
            optim.step()
            if ema is not None:
                ema.update(model_for_state)
                if ema.num_updates != global_step + 1:
                    raise RuntimeError(
                        f"EMA/optimizer update mismatch: ema={ema.num_updates}, expected={global_step + 1}"
                    )
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            update_loss = accumulated_loss
            reduced_metrics = reduce_scalar_metrics(
                {
                    "loss": update_loss,
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    **({"ema_updates": float(ema.num_updates)} if ema is not None else {}),
                    **accumulated_auxiliary_values,
                },
                device,
            )
            update_loss = reduced_metrics["loss"]
            if is_main:
                infos.append(
                    {
                        **reduced_metrics,
                        "learning_rate": optim.param_groups[0]["lr"],
                    }
                )
            micro_step_in_update = 0
            accumulated_loss = 0.0
            accumulated_auxiliary_values = {}

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time
                logged_updates = len(infos)
                steps_per_second = logged_updates / max(elapsed, 1e-9)
                samples_per_second = effective_global_batch * steps_per_second

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                auxiliary_averages = {
                    key: sum(info[key] for info in infos if key in info) / sum(key in info for info in infos)
                    for key in sorted({key for info in infos for key in info})
                    if key not in {"loss", "learning_rate", "grad_norm"}
                }

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} "
                    f"steps/s={steps_per_second:.4f} samples/s={samples_per_second:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} "
                    f"steps/s={steps_per_second:.4f} samples/s={samples_per_second:.2f} time={elapsed:.1f}s"
                )
                if auxiliary_averages:
                    logging.info(
                        "Auxiliary metrics: "
                        + " ".join(f"{key}={value:.5f}" for key, value in auxiliary_averages.items())
                    )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / logged_updates,
                        "steps_per_second": steps_per_second,
                        "samples_per_second": samples_per_second,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    log_payload.update(auxiliary_averages)
                    wandb.log(log_payload, step=global_step)

                if torch.cuda.is_available():
                    log_memory_usage(device, global_step, "logging")

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(
                model,
                optim,
                global_step,
                config,
                is_main,
                data_config,
                loader,
                ema,
                micro_step_in_update=micro_step_in_update,
            )
            if use_ddp and should_save_checkpoint(global_step, config):
                # Keep non-writing ranks from entering the next forward while
                # rank 0 is still writing a large atomic checkpoint.
                dist.barrier()

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{update_loss:.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
