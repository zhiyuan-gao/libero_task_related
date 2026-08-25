from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import random
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.training.policy_aux_dataset as _policy_aux_dataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IndexedSubsetDataset(Dataset[T_co]):
    """Read an immutable subset while preserving the base dataset's episode index table."""

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]):
        self._dataset = dataset
        self._indices = tuple(int(index) for index in indices)
        if not self._indices or len(set(self._indices)) != len(self._indices):
            raise ValueError("Subset indices must be non-empty and unique")
        if min(self._indices) < 0 or max(self._indices) >= len(dataset):
            raise IndexError("Subset index is outside the base dataset")

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._dataset[self._indices[index.__index__()]]

    def __len__(self) -> int:
        return len(self._indices)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    policy_aux_config: _policy_aux_dataset.PolicyAuxTrainConfig | None = None,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    episodes = None if data_config.lerobot_episodes is None else list(data_config.lerobot_episodes)
    revision = data_config.lerobot_revision
    root = data_config.lerobot_root
    subset_dataset_indices = None
    if policy_aux_config is not None:
        aux_episodes = policy_aux_config.lerobot_episode_indices()
        aux_revision = policy_aux_config.lerobot_revision
        aux_root = policy_aux_config.lerobot_root
        if episodes is not None and episodes != aux_episodes:
            raise ValueError("DataConfig and policy_aux select different LeRobot episodes")
        if revision is not None and revision != aux_revision:
            raise ValueError("DataConfig and policy_aux select different LeRobot revisions")
        if root is not None and root != aux_root:
            raise ValueError("DataConfig and policy_aux select different LeRobot roots")
        episodes, revision, root = aux_episodes, aux_revision, aux_root
        if policy_aux_config.lerobot_task_indices is not None:
            # LeRobot v2.0 builds a compact episode_data_index for selected
            # episodes but indexes it with original, non-contiguous episode
            # IDs. Load the canonical contiguous LIBERO-10 population so its
            # action-chunk boundaries stay valid, then expose only approved
            # frame identities through a read-only wrapper.
            episodes = list(range(_policy_aux_dataset.CANONICAL_LIBERO_EPISODES))
            subset_dataset_indices = policy_aux_config.lerobot_dataset_indices()
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=root,
        episodes=episodes,
        revision=revision,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )
    if subset_dataset_indices is not None:
        dataset = IndexedSubsetDataset(dataset, subset_dataset_indices)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        policy_aux_config=config.policy_aux,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    policy_aux_config: _policy_aux_dataset.PolicyAuxTrainConfig | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config, policy_aux_config=policy_aux_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    if policy_aux_config is not None:
        if framework != "pytorch":
            raise ValueError("Policy auxiliary training is implemented only for the PyTorch path")
        dataset = _policy_aux_dataset.PolicyAuxTransformedDataset(dataset, policy_aux_config)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires optional DROID-specific dependencies.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        # Avoid initializing the JAX/XLA CUDA backend inside PyTorch DDP
        # processes. Besides being unnecessary, one XLA context per rank can
        # conflict with rank-local CUDA checkpoint restoration.
        if framework == "jax" and jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._sampler = sampler
        self._epoch = 0
        self._batch_in_epoch = 0
        self._total_batches_yielded = 0
        self._epoch_start_generator_state = None
        self._resume_pending = False

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._generator = generator
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            # Recreate workers at every epoch boundary. Their RNG seeds are
            # derived from ``generator``, whose epoch-start state is captured
            # below; this makes replay after an exact resume deterministic.
            persistent_workers=False,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def state_dict(self) -> dict:
        """Capture the exact epoch and batch position for trajectory-exact resume."""

        epoch_start = self._epoch_start_generator_state
        if epoch_start is None:
            epoch_start = self._generator.get_state()
        sampler_state = None
        if isinstance(self._sampler, torch.utils.data.distributed.DistributedSampler):
            sampler_state = {
                "type": type(self._sampler).__qualname__,
                "num_replicas": self._sampler.num_replicas,
                "rank": self._sampler.rank,
                "shuffle": self._sampler.shuffle,
                "seed": self._sampler.seed,
                "drop_last": self._sampler.drop_last,
            }
        return {
            "schema": "openpi.torch_data_loader_state.v1",
            "epoch": self._epoch,
            "batch_in_epoch": self._batch_in_epoch,
            "total_batches_yielded": self._total_batches_yielded,
            "epoch_start_generator_state": epoch_start.clone(),
            "epoch_length": len(self._data_loader),
            "dataset_length": len(self._data_loader.dataset),
            "batch_size": self._data_loader.batch_size,
            "drop_last": self._data_loader.drop_last,
            "distributed_sampler": sampler_state,
        }

    def load_state_dict(self, state: dict) -> None:
        """Schedule a deterministic replay to the saved within-epoch position."""

        if state.get("schema") != "openpi.torch_data_loader_state.v1":
            raise ValueError(f"Unsupported data-loader resume state: {state.get('schema')}")
        epoch = int(state["epoch"])
        batch_in_epoch = int(state["batch_in_epoch"])
        total_batches_yielded = int(state["total_batches_yielded"])
        if epoch < 0 or batch_in_epoch < 0 or total_batches_yielded < 0:
            raise ValueError("Data-loader resume counters must be non-negative")
        current_state = self.state_dict()
        for name in ("epoch_length", "dataset_length", "batch_size", "drop_last", "distributed_sampler"):
            if state[name] != current_state[name]:
                raise ValueError(
                    f"Exact resume data-loader mismatch for {name}: saved={state[name]}, current={current_state[name]}"
                )
        if batch_in_epoch > len(self._data_loader):
            raise ValueError(f"Saved batch offset {batch_in_epoch} exceeds epoch length {len(self._data_loader)}")
        generator_state = state["epoch_start_generator_state"]
        if not isinstance(generator_state, torch.Tensor) or generator_state.device.type != "cpu":
            raise ValueError("Data-loader generator state must be a CPU tensor")
        self._epoch = epoch
        self._batch_in_epoch = batch_in_epoch
        self._total_batches_yielded = total_batches_yielded
        self._epoch_start_generator_state = generator_state.clone()
        self._resume_pending = True

    def __iter__(self):
        while True:
            if self._num_batches is not None and self._total_batches_yielded >= self._num_batches:
                return

            skip_batches = 0
            if self._resume_pending:
                self._generator.set_state(self._epoch_start_generator_state)
                skip_batches = self._batch_in_epoch
                self._resume_pending = False
            else:
                self._batch_in_epoch = 0
                self._epoch_start_generator_state = self._generator.get_state().clone()
            if isinstance(self._sampler, torch.utils.data.distributed.DistributedSampler):
                self._sampler.set_epoch(self._epoch)
            data_iter = iter(self._data_loader)
            # Replaying already-consumed batches restores the sampler/worker
            # position. Preserve main-process RNG in case a dataset transform
            # happens to use Python, NumPy, or torch randomness while replaying.
            replay_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            try:
                for _ in range(skip_batches):
                    try:
                        next(data_iter)
                    except StopIteration as error:
                        raise RuntimeError(
                            f"Cannot restore batch {skip_batches} within data-loader epoch {self._epoch}"
                        ) from error
            finally:
                random.setstate(replay_rng[0])
                np.random.set_state(replay_rng[1])
                torch.set_rng_state(replay_rng[2])
            while True:
                if self._num_batches is not None and self._total_batches_yielded >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self._epoch += 1
                    self._batch_in_epoch = 0
                    self._epoch_start_generator_state = None
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                self._batch_in_epoch += 1
                self._total_batches_yielded += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def state_dict(self) -> dict:
        if not isinstance(self._data_loader, TorchDataLoader):
            raise TypeError("Exact resume state is available only for the PyTorch TorchDataLoader")
        return self._data_loader.state_dict()

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(self._data_loader, TorchDataLoader):
            raise TypeError("Exact resume state is available only for the PyTorch TorchDataLoader")
        self._data_loader.load_state_dict(state)

    def __iter__(self):
        for batch in self._data_loader:
            observation = _model.Observation.from_dict(batch)
            if "policy_aux" in batch:
                yield observation, batch["actions"], batch["policy_aux"]
            else:
                yield observation, batch["actions"]
