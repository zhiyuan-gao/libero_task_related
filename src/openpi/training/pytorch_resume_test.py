from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from openpi.training import data_loader as _data_loader
from openpi.training import pytorch_resume


class _IndexedRegressionDataset:
    def __len__(self) -> int:
        return 24

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        value = np.asarray([float(index)], dtype=np.float32)
        return {"identity": np.asarray(index, dtype=np.int64), "input": value, "target": 3.0 * value - 1.0}


class _WorkerRandomDataset:
    def __len__(self) -> int:
        return 24

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {
            "identity": np.asarray(index, dtype=np.int64),
            "numpy": np.asarray(np.random.random(), dtype=np.float64),
            "python": np.asarray(random.random(), dtype=np.float64),
            "torch": np.asarray(torch.rand(()).item(), dtype=np.float64),
        }


def _components(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = _IndexedRegressionDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        drop_last=True,
    )
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        sampler=sampler,
        seed=seed,
        framework="pytorch",
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return loader, model, optimizer


def _updates(loader, model, optimizer, count: int) -> tuple[list[tuple[int, ...]], list[float]]:
    identities = []
    losses = []
    batches = iter(loader)
    for _ in range(count):
        batch = next(batches)
        identities.append(tuple(int(value) for value in batch["identity"]))
        stochastic_offset = torch.rand(()) + float(np.random.uniform()) + random.random()
        prediction = model(batch["input"].float())
        loss = torch.nn.functional.mse_loss(prediction + stochastic_offset * 1e-3, batch["target"].float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return identities, losses


def test_uninterrupted_and_resumed_updates_are_trajectory_exact(tmp_path) -> None:
    seed = 731
    total_updates = 10
    split_update = 7

    full_loader, full_model, full_optimizer = _components(seed)
    full_identities, full_losses = _updates(full_loader, full_model, full_optimizer, total_updates)

    split_loader, split_model, split_optimizer = _components(seed)
    split_identities, split_losses = _updates(split_loader, split_model, split_optimizer, split_update)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": split_model.state_dict(),
            "optimizer": split_optimizer.state_dict(),
            "training": pytorch_resume.capture_training_state(
                split_loader,
                micro_step_in_update=0,
                rank=0,
                world_size=1,
            ),
        },
        checkpoint_path,
    )

    resumed_loader, resumed_model, resumed_optimizer = _components(seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    resumed_model.load_state_dict(checkpoint["model"], strict=True)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    pytorch_resume.restore_training_state(checkpoint["training"], resumed_loader, rank=0, world_size=1)
    resumed_identities, resumed_losses = _updates(
        resumed_loader,
        resumed_model,
        resumed_optimizer,
        total_updates - split_update,
    )

    assert split_identities + resumed_identities == full_identities
    assert resumed_identities[0] == full_identities[split_update]
    assert split_losses + resumed_losses == pytest.approx(full_losses, rel=0.0, abs=0.0)
    for resumed, uninterrupted in zip(resumed_model.parameters(), full_model.parameters(), strict=True):
        assert torch.equal(resumed, uninterrupted)


def test_checkpoint_rejects_partial_gradient_accumulation() -> None:
    loader, _, _ = _components(17)
    with pytest.raises(ValueError, match="gradient-accumulation boundary"):
        pytorch_resume.capture_training_state(loader, micro_step_in_update=1, rank=0, world_size=1)


def _worker_random_loader(seed: int) -> _data_loader.TorchDataLoader:
    dataset = _WorkerRandomDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )
    return _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        sampler=sampler,
        num_workers=2,
        seed=seed,
        framework="pytorch",
    )


def _collect_worker_batches(loader, count: int) -> list[dict[str, np.ndarray]]:
    batches = iter(loader)
    return [{key: np.asarray(value).copy() for key, value in next(batches).items()} for _ in range(count)]


def test_multiworker_random_transforms_resume_exactly_across_epochs() -> None:
    seed = 4821
    split_batch = 8  # Six batches per epoch, so checkpoint after an epoch boundary.
    total_batches = 12

    uninterrupted_loader = _worker_random_loader(seed)
    uninterrupted = _collect_worker_batches(uninterrupted_loader, total_batches)

    split_loader = _worker_random_loader(seed)
    before_resume = _collect_worker_batches(split_loader, split_batch)
    state = pytorch_resume.capture_training_state(
        split_loader,
        micro_step_in_update=0,
        rank=0,
        world_size=1,
    )

    resumed_loader = _worker_random_loader(seed)
    pytorch_resume.restore_training_state(state, resumed_loader, rank=0, world_size=1)
    after_resume = _collect_worker_batches(resumed_loader, total_batches - split_batch)

    candidate = before_resume + after_resume
    for candidate_batch, expected_batch in zip(candidate, uninterrupted, strict=True):
        assert candidate_batch.keys() == expected_batch.keys()
        for key in candidate_batch:
            np.testing.assert_array_equal(candidate_batch[key], expected_batch[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_local_cuda_rng_state_is_restored() -> None:
    loader, _, _ = _components(29)
    torch.cuda.manual_seed(991)
    state = pytorch_resume.capture_training_state(
        loader,
        micro_step_in_update=0,
        rank=0,
        world_size=1,
    )
    expected = torch.rand(16, device="cuda")
    _ = torch.rand(16, device="cuda")
    pytorch_resume.restore_training_state(state, loader, rank=0, world_size=1)
    candidate = torch.rand(16, device="cuda")
    assert torch.equal(candidate, expected)
