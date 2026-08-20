from __future__ import annotations

from types import SimpleNamespace

import torch

from openpi.models_pytorch import preprocessing_pytorch
from openpi.models_pytorch.auxiliary_heads import grounding_focal_dice_loss
from openpi.models_pytorch.auxiliary_heads import masked_standardized_mse
from openpi.models_pytorch.auxiliary_heads import masked_standardized_smooth_l1
from openpi.models_pytorch.pi05_aux_queries import JointP2TrainLayout
from openpi.models_pytorch.pi05_aux_queries import PrefixLayout
from openpi.models_pytorch.pi05_aux_queries import TokenSpan
from openpi.models_pytorch.pi05_aux_queries import build_explicit_aux_prefix_attention
from openpi.models_pytorch.pi05_aux_queries import build_explicit_aux_train_attention
from openpi.models_pytorch.pi05_aux_queries import build_joint_p2_attention
from openpi.models_pytorch.pi05_aux_queries import build_joint_p2_position_ids
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
            num_ground_queries=0 if mode in ("semantic_geometry", "semantic_geometry_motion") else 8,
            num_geometry_queries=8,
            num_motion_queries=8 if mode == "semantic_geometry_motion" else 0,
            ground_mask_dim=256,
            ground_focal_alpha=0.25,
            ground_focal_gamma=2.0,
            lambda_geo=0.15,
            lambda_ground=0.50 if mode == "ground_geometry_semantic_lm" else None,
            lambda_sem=0.01
            if mode in ("semantic_geometry", "semantic_geometry_motion", "ground_geometry_semantic_lm")
            else None,
            lambda_motion=0.10 if mode == "semantic_geometry_motion" else None,
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
    semantic_geometry = create_pytorch_model(_train_config("semantic_geometry"))
    semantic_geometry_motion = create_pytorch_model(_train_config("semantic_geometry_motion"))
    p2 = create_pytorch_model(_train_config("ground_geometry_semantic_lm"))

    assert type(plain) is FakePlain
    assert type(p1) is FakeAux
    assert type(p2) is FakeAux
    assert p1.aux_config.mode == "geometry"
    assert semantic_geometry.aux_config.mode == "semantic_geometry"
    assert semantic_geometry.aux_config.num_ground_queries == 0
    assert semantic_geometry_motion.aux_config.mode == "semantic_geometry_motion"
    assert semantic_geometry_motion.aux_config.num_motion_queries == 8
    assert p2.aux_config.mode == "ground_geometry_semantic_lm"
    for model in (p1, semantic_geometry, semantic_geometry_motion, p2):
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


def test_semantic_geometry_layout_has_geometry_and_no_ground() -> None:
    layout = PrefixLayout(
        view_spans={"agent": TokenSpan(0, 2), "wrist": TokenSpan(2, 4)},
        real_view_names=("agent", "wrist"),
        padded_view_names=(),
        language=TokenSpan(4, 6),
        context=TokenSpan(0, 6),
        geometry=TokenSpan(6, 14),
        ground=None,
        action_suffix=TokenSpan(14, 17),
    )
    prefix_pad = torch.ones((1, 14), dtype=torch.bool)
    suffix_pad = torch.ones((1, 3), dtype=torch.bool)
    suffix_ar = torch.tensor([[1, 0, 0]], dtype=torch.bool)
    mask = build_explicit_aux_train_attention(prefix_pad, suffix_pad, suffix_ar, layout)[0]

    assert layout.query_groups == {"geometry": TokenSpan(6, 14)}
    assert layout.ground is None
    assert bool(mask[14:, 6:14].all()) is True
    assert bool(mask[6:14, 14:].any()) is False


def test_b_motion_queries_are_isolated_and_visible_to_action() -> None:
    layout = PrefixLayout(
        view_spans={"agent": TokenSpan(0, 2)},
        real_view_names=("agent",),
        padded_view_names=(),
        language=TokenSpan(2, 4),
        context=TokenSpan(0, 4),
        geometry=TokenSpan(4, 12),
        ground=None,
        motion=TokenSpan(12, 20),
        action_suffix=TokenSpan(20, 23),
    )
    prefix_pad = torch.ones((1, 20), dtype=torch.bool)
    suffix_pad = torch.ones((1, 3), dtype=torch.bool)
    suffix_ar = torch.tensor([[1, 0, 0]], dtype=torch.bool)
    mask = build_explicit_aux_train_attention(prefix_pad, suffix_pad, suffix_ar, layout)[0]
    assert layout.query_groups == {
        "geometry": TokenSpan(4, 12),
        "motion": TokenSpan(12, 20),
    }
    assert not bool(mask[4:12, 12:20].any())
    assert not bool(mask[12:20, 4:12].any())
    assert bool(mask[20:, :20].all())


def test_motion_smooth_l1_masks_invalid_samples_exactly() -> None:
    prediction = torch.tensor([[2.0, 0.0], [100.0, -100.0]], requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = masked_standardized_smooth_l1(prediction, target, torch.tensor([True, False]), torch.zeros(2), torch.ones(2))
    assert torch.isclose(loss, torch.tensor(0.75))
    loss.backward()
    assert torch.equal(prediction.grad[1], torch.zeros(2))


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


def _joint_p2_layout() -> JointP2TrainLayout:
    return JointP2TrainLayout(
        base_layout=_p2_layout(),
        semantic=TokenSpan(10, 13),
        action_suffix=TokenSpan(13, 16),
    )


def test_joint_p2_attention_has_exact_branch_connectivity_and_padding() -> None:
    layout = _joint_p2_layout()
    paligemma_pad = torch.tensor([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    suffix_pad = torch.ones((1, 3), dtype=torch.bool)
    suffix_ar = torch.tensor([[1, 0, 0]], dtype=torch.bool)
    mask = build_joint_p2_attention(paligemma_pad, suffix_pad, suffix_ar, layout)[0]

    context = slice(0, 6)
    geometry = slice(6, 8)
    ground = slice(8, 10)
    semantic = slice(10, 13)
    action = slice(13, 16)

    assert bool(mask[context, geometry].any()) is False
    assert bool(mask[context, ground].any()) is False
    assert bool(mask[context, semantic].any()) is False
    assert bool(mask[context, action].any()) is False
    assert bool(mask[geometry, :5].all()) is True
    assert bool(mask[geometry, geometry].all()) is True
    assert bool(mask[geometry, ground].any()) is False
    assert bool(mask[geometry, semantic].any()) is False
    assert bool(mask[ground, :5].all()) is True
    assert bool(mask[ground, ground].all()) is True
    assert bool(mask[ground, geometry].any()) is False
    assert bool(mask[ground, semantic].any()) is False
    assert bool(mask[semantic, geometry].any()) is False
    assert bool(mask[semantic, ground].any()) is False
    assert bool(mask[semantic, action].any()) is False
    assert bool(mask[10, :5].all()) is True
    assert bool(mask[10, 10]) is True
    assert bool(mask[10, 11]) is False
    assert bool(mask[11, 10:12].all()) is True
    assert bool(mask[action, :5].all()) is True
    assert bool(mask[action, geometry].all()) is True
    assert bool(mask[action, ground].all()) is True
    assert bool(mask[action, semantic].any()) is False
    assert bool(mask[action, action].all()) is True
    for padded_index in (5, 12):
        assert bool(mask[padded_index].any()) is False
        assert bool(mask[:, padded_index].any()) is False


def test_semantic_geometry_uses_p2_joint_mask_without_ground() -> None:
    base_layout = PrefixLayout(
        view_spans={"agent": TokenSpan(0, 2), "wrist": TokenSpan(2, 4)},
        real_view_names=("agent", "wrist"),
        padded_view_names=(),
        language=TokenSpan(4, 6),
        context=TokenSpan(0, 6),
        geometry=TokenSpan(6, 8),
        ground=None,
    )
    layout = JointP2TrainLayout(
        base_layout=base_layout,
        semantic=TokenSpan(8, 11),
        action_suffix=TokenSpan(11, 14),
    )
    paligemma_pad = torch.ones((1, 11), dtype=torch.bool)
    suffix_pad = torch.ones((1, 3), dtype=torch.bool)
    suffix_ar = torch.tensor([[1, 0, 0]], dtype=torch.bool)
    mask = build_joint_p2_attention(paligemma_pad, suffix_pad, suffix_ar, layout)[0]

    # Geometry follows the P2 isolated-query rule and Semantic reads only
    # Context plus its causal teacher prefix.
    assert bool(mask[6:8, :6].all()) is True
    assert bool(mask[6:8, 6:8].all()) is True
    assert bool(mask[6:8, 8:].any()) is False
    assert bool(mask[8:11, :6].all()) is True
    assert bool(mask[8:11, 6:8].any()) is False

    # The Action suffix reads Context and Geometry exactly as P2 does after
    # deleting Ground, while Semantic teacher tokens remain structurally hidden.
    assert bool(mask[11:14, :8].all()) is True
    assert bool(mask[11:14, 8:11].any()) is False


def test_joint_p2_positions_match_old_main_and_semantic_references() -> None:
    layout = _joint_p2_layout()
    base_pad = torch.tensor([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.bool)
    semantic_pad = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    suffix_pad = torch.ones((1, 3), dtype=torch.bool)
    merged = build_joint_p2_position_ids(base_pad, semantic_pad, suffix_pad, layout)

    old_main = torch.cumsum(torch.cat((base_pad, suffix_pad), dim=1), dim=1) - 1
    old_semantic = (
        torch.cumsum(torch.cat((base_pad[:, : layout.base_layout.context.end], semantic_pad), dim=1), dim=1) - 1
    ).clamp_min(0)
    assert torch.equal(merged[:, : layout.semantic.start], old_main[:, : layout.semantic.start])
    assert torch.equal(
        merged[:, layout.semantic.start : layout.semantic.end],
        old_semantic[:, layout.base_layout.context.end :],
    )
    assert torch.equal(
        merged[:, layout.action_suffix.start : layout.action_suffix.end],
        old_main[:, layout.semantic.start :],
    )


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
