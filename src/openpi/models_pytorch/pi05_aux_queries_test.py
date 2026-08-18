from __future__ import annotations

from types import SimpleNamespace

import torch

from openpi.models_pytorch import preprocessing_pytorch
from openpi.models_pytorch.auxiliary_heads import grounding_focal_dice_loss
from openpi.models_pytorch.auxiliary_heads import masked_standardized_mse
from openpi.models_pytorch.pi05_aux_queries import PrefixLayout
from openpi.models_pytorch.pi05_aux_queries import TokenSpan
from openpi.models_pytorch.pi05_aux_queries import build_explicit_aux_prefix_attention
from openpi.models_pytorch.pi05_aux_queries import build_explicit_aux_train_attention
from openpi.models_pytorch.pi05_aux_queries import build_native_semantic_lm_attention
from openpi.models_pytorch.pi05_aux_queries import create_pytorch_model
from openpi.models_pytorch.pi05_aux_queries import load_trained_pytorch_model
from openpi.models_pytorch.policy_aux_preprocessing import patch_foreground_coverage
from openpi.models_pytorch.policy_aux_preprocessing import preprocess_observation_and_ground_masks_pytorch
from openpi.models_pytorch.policy_aux_preprocessing import raw_opengl_mask_to_policy_canvas


def _train_config(mode: str | None):
    policy_aux = None
    if mode is not None:
        policy_aux = SimpleNamespace(
            mode=mode,
            num_ground_queries=8,
            num_geometry_queries=8,
            ground_mask_dim=256,
            ground_focal_alpha=0.25,
            ground_focal_gamma=2.0,
            lambda_geo=0.15,
            lambda_ground=0.50 if mode == "ground_geometry_semantic_lm" else None,
            lambda_sem=0.01 if mode == "ground_geometry_semantic_lm" else None,
            policy_manifest_path="/unusable/semantic-and-grounding",
            geometry_target_index_path="/unusable/geometry",
        )
    return SimpleNamespace(model=object(), policy_aux=policy_aux)


def test_shared_factory_selects_plain_or_aux_and_drops_teacher_paths(monkeypatch) -> None:
    class FakePlain:
        def __init__(self, config) -> None:
            self.config = config

    class FakeAux(FakePlain):
        def __init__(self, config, aux_config) -> None:
            super().__init__(config)
            self.aux_config = aux_config

    monkeypatch.setattr("openpi.models_pytorch.pi05_aux_queries.PI0Pytorch", FakePlain)
    monkeypatch.setattr("openpi.models_pytorch.pi05_aux_queries.PI05AuxPolicy", FakeAux)

    plain = create_pytorch_model(_train_config(None))
    p1 = create_pytorch_model(_train_config("geometry"))
    p2 = create_pytorch_model(_train_config("ground_geometry_semantic_lm"))

    assert type(plain) is FakePlain
    assert type(p1) is FakeAux
    assert type(p2) is FakeAux
    assert p1.aux_config.mode == "geometry"
    assert p2.aux_config.mode == "ground_geometry_semantic_lm"
    for model in (p1, p2):
        assert model.aux_config.semantic_annotation_root is None
        assert model.aux_config.ground_mask_root is None
        assert model.aux_config.geometry_cache_root is None
        assert model.aux_config.geometry_normalization_path is None


def test_trained_checkpoint_loader_is_strict(monkeypatch) -> None:
    model = object()
    observed = {}
    monkeypatch.setattr("openpi.models_pytorch.pi05_aux_queries.create_pytorch_model", lambda _: model)

    def fake_load(loaded_model, weight_path, *, strict, device):
        observed.update(model=loaded_model, weight_path=weight_path, strict=strict, device=device)
        return set(), []

    monkeypatch.setattr("openpi.models_pytorch.pi05_aux_queries.safetensors.torch.load_model", fake_load)
    assert load_trained_pytorch_model(_train_config("geometry"), "/checkpoint/model.safetensors") is model
    assert observed == {
        "model": model,
        "weight_path": "/checkpoint/model.safetensors",
        "strict": True,
        "device": "cpu",
    }


def _p2_layout() -> PrefixLayout:
    return PrefixLayout(
        view_spans={"agent": TokenSpan(0, 2), "wrist": TokenSpan(2, 4)},
        real_view_names=("agent", "wrist"),
        padded_view_names=("right_wrist",),
        language=TokenSpan(4, 6),
        context=TokenSpan(0, 6),
        geometry=TokenSpan(6, 8),
        ground=TokenSpan(8, 10),
    )


def test_explicit_p2_attention_rectangles() -> None:
    layout = _p2_layout()
    pad = torch.tensor([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.bool)
    mask = build_explicit_aux_prefix_attention(pad, layout)[0]

    context = slice(layout.context.start, layout.context.end)
    ground = slice(layout.ground.start, layout.ground.end)
    geometry = slice(layout.geometry.start, layout.geometry.end)

    assert bool(mask[context, ground].any()) is False
    assert bool(mask[context, geometry].any()) is False
    assert bool(mask[ground, context][:, :5].all()) is True
    assert bool(mask[geometry, context][:, :5].all()) is True
    assert bool(mask[ground, ground].all()) is True
    assert bool(mask[geometry, geometry].all()) is True
    assert bool(mask[ground, geometry].any()) is False
    assert bool(mask[geometry, ground].any()) is False
    assert bool(mask[:, 5].any()) is False
    assert bool(mask[5, :].any()) is False


def test_action_suffix_reads_all_enabled_groups_and_prefix_cannot_read_action() -> None:
    layout = _p2_layout()
    prefix_pad = torch.ones((1, 10), dtype=torch.bool)
    suffix_pad = torch.ones((1, 3), dtype=torch.bool)
    suffix_ar = torch.tensor([[1, 0, 0]], dtype=torch.bool)
    mask = build_explicit_aux_train_attention(prefix_pad, suffix_pad, suffix_ar, layout)[0]

    assert bool(mask[:10, 10:].any()) is False
    assert bool(mask[10:, :10].all()) is True
    assert bool(mask[10:, 10:].all()) is True


def test_native_semantic_lm_attention_is_separate_prefix_lm() -> None:
    context_pad = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    teacher_pad = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    mask = build_native_semantic_lm_attention(context_pad, teacher_pad)[0]

    context = slice(0, 4)
    teacher = slice(4, 7)
    assert bool(mask[context, teacher].any()) is False
    assert bool(mask[:3, :3].all()) is True
    assert bool(mask[4, :3].all()) is True
    assert bool(mask[5, :3].all()) is True
    assert bool(mask[4, 4]) is True
    assert bool(mask[4, 5]) is False
    assert bool(mask[5, 4:6].all()) is True
    assert bool(mask[3].any()) is False
    assert bool(mask[:, 3].any()) is False
    assert bool(mask[6].any()) is False
    assert bool(mask[:, 6].any()) is False


def test_geometry_invalid_samples_do_not_contribute() -> None:
    prediction = torch.tensor([[1.0, 2.0], [10.0, 10.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0], [-100.0, -100.0]])
    loss = masked_standardized_mse(
        prediction,
        target,
        torch.tensor([True, False]),
        torch.zeros(2),
        torch.ones(2),
    )
    assert torch.equal(loss.detach(), torch.tensor(2.5))
    loss.backward()
    assert torch.equal(prediction.grad[1], torch.zeros(2))


def test_grounding_invalid_views_are_ignored() -> None:
    logits = torch.zeros((1, 2, 4), requires_grad=True)
    target = torch.tensor([[[1.0, 0.0, 0.5, 0.0], [1.0, 1.0, 1.0, 1.0]]])
    result = grounding_focal_dice_loss(logits, target, torch.tensor([[True, False]]))
    assert result["valid_view_count"].item() == 1
    assert torch.equal(result["valid_count_by_view"], torch.tensor([1, 0]))
    assert result["dice_score_by_view"][0].item() > 0
    assert result["dice_score_by_view"][1].item() == 0
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert torch.equal(logits.grad[0, 1], torch.zeros(4))


class _Observation:
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(11)
        self.images = {
            name: torch.rand((2, 3, 224, 224), generator=generator) * 2 - 1
            for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        }
        self.image_masks = {
            "base_0_rgb": torch.ones(2, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(2, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(2, dtype=torch.bool),
        }
        self.state = torch.zeros((2, 32))
        self.tokenized_prompt = torch.ones((2, 8), dtype=torch.long)
        self.tokenized_prompt_mask = torch.ones((2, 8), dtype=torch.bool)
        self.token_ar_mask = torch.zeros((2, 8), dtype=torch.bool)
        self.token_loss_mask = torch.zeros((2, 8), dtype=torch.bool)


def test_synchronized_preprocessing_preserves_official_rgb_path() -> None:
    observation = _Observation()
    masks = {
        "base_0_rgb": torch.zeros((2, 224, 224)),
        "left_wrist_0_rgb": torch.zeros((2, 224, 224)),
    }
    torch.manual_seed(37)
    official = preprocessing_pytorch.preprocess_observation_pytorch(observation, train=True)
    torch.manual_seed(37)
    synchronized = preprocess_observation_and_ground_masks_pytorch(observation, masks, train=True)
    for name in observation.images:
        assert torch.equal(official.images[name], synchronized.observation.images[name])


def test_raw_mask_rotation_and_patch_coverage() -> None:
    raw = torch.zeros((4, 4))
    raw[0, 0] = 1.0
    policy = raw_opengl_mask_to_policy_canvas(raw, image_resolution=(4, 4))
    assert policy[3, 3].item() == 1.0
    assert policy.sum().item() == 1.0
    patches = patch_foreground_coverage(policy[None], grid_height=2, grid_width=2)
    assert torch.equal(patches, torch.tensor([[0.0, 0.0, 0.0, 0.25]]))
