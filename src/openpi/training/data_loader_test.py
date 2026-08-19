import dataclasses
import math

import jax
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
        framework="pytorch",
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, framework="pytorch")
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_distributed_sampler_epoch_advances_when_infinite_loader_restarts():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    class RecordingSampler(torch.utils.data.distributed.DistributedSampler):
        def __init__(self):
            super().__init__(dataset, num_replicas=2, rank=0, shuffle=True, drop_last=True)
            self.epochs = []

        def set_epoch(self, epoch: int) -> None:
            self.epochs.append(epoch)
            super().set_epoch(epoch)

    sampler = RecordingSampler()
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=2,
        sampler=sampler,
        num_batches=3,
        framework="pytorch",
    )
    assert len(list(loader)) == 3
    assert sampler.epochs == [0, 1, 2]


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
        num_workers=2,
        framework="pytorch",
    )
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")
    config = dataclasses.replace(config, batch_size=math.lcm(config.batch_size, jax.device_count()))

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=math.lcm(4, jax.device_count()))

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
