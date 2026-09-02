"""Deferred CPU checks for the written ablation topology; not auto-executed."""

from __future__ import annotations

from robocasa24_finetune.ablations.configs import ABLATION_KEEP_PERIOD
from robocasa24_finetune.ablations.configs import ABLATION_SAVE_INTERVAL
from robocasa24_finetune.ablations.model import AblationPrefixLayout
from robocasa24_finetune.ablations.model import build_ablation_joint_attention
from robocasa24_finetune.ablations.model import build_ablation_joint_position_ids
from robocasa24_finetune.ablations.specs import ABLATION_SPECS
import torch

from openpi.models_pytorch import pi05_aux_queries as aux_model


def _layout(action_conditioning: tuple[str, ...]) -> aux_model.JointP2TrainLayout:
    base = AblationPrefixLayout(
        view_spans={"agent": aux_model.TokenSpan(0, 2)},
        real_view_names=("agent",),
        padded_view_names=(),
        language=aux_model.TokenSpan(2, 4),
        context=aux_model.TokenSpan(0, 4),
        ground=None,
        geometry=aux_model.TokenSpan(4, 6),
        motion=aux_model.TokenSpan(6, 8),
        action_conditioning=action_conditioning,
    )
    return aux_model.JointP2TrainLayout(
        base_layout=base,
        semantic=aux_model.TokenSpan(8, 10),
        action_suffix=aux_model.TokenSpan(10, 13),
    )


def test_six_ablation_specs_are_complete_and_matched() -> None:
    assert tuple(ABLATION_SPECS) == (
        "geometry_only",
        "semantic_geometry",
        "semantic_motion",
        "full",
        "supervision_only",
        "whole_scene",
    )
    assert ABLATION_SPECS["semantic_motion"].geometry_enabled is False
    assert ABLATION_SPECS["semantic_motion"].action_conditioning == ("motion",)
    assert ABLATION_SPECS["supervision_only"].action_conditioning == ()
    assert ABLATION_SPECS["whole_scene"].target_scope == "whole_scene"


def test_ablation_checkpoint_retention_matches_periodic_style() -> None:
    assert ABLATION_SAVE_INTERVAL == 1_000
    assert ABLATION_KEEP_PERIOD == 5_000
    scheduled = list(range(ABLATION_SAVE_INTERVAL, 28_001, ABLATION_SAVE_INTERVAL))
    protected = {step for step in scheduled if step % ABLATION_KEEP_PERIOD == 0}
    retained = protected | {scheduled[-1]}
    expected = {5_000, 10_000, 15_000, 20_000, 25_000, 28_000}
    assert retained == expected


def test_full_action_reads_both_queries_but_not_semantic_teacher() -> None:
    layout = _layout(("geometry", "motion"))
    prefix_pad = torch.ones((1, 10), dtype=torch.bool)
    suffix_pad = torch.ones((1, 3), dtype=torch.bool)
    suffix_ar = torch.tensor([[True, False, False]])
    mask = build_ablation_joint_attention(
        prefix_pad,
        suffix_pad,
        suffix_ar,
        layout,
    )[0]
    action = slice(10, 13)
    assert bool(mask[action, :8].all()) is True
    assert bool(mask[action, 8:10].any()) is False


def test_supervision_only_action_reads_native_context_and_no_queries() -> None:
    layout = _layout(())
    prefix_pad = torch.ones((1, 10), dtype=torch.bool)
    suffix_pad = torch.ones((1, 3), dtype=torch.bool)
    suffix_ar = torch.tensor([[True, False, False]])
    mask = build_ablation_joint_attention(
        prefix_pad,
        suffix_pad,
        suffix_ar,
        layout,
    )[0]
    action = slice(10, 13)
    assert bool(mask[action, :4].all()) is True
    assert bool(mask[action, 4:8].any()) is False
    assert bool(mask[action, 8:10].any()) is False


def test_supervision_only_action_positions_match_native_context_length() -> None:
    layout = _layout(())
    base_pad = torch.ones((1, 8), dtype=torch.bool)
    semantic_pad = torch.ones((1, 2), dtype=torch.bool)
    suffix_pad = torch.ones((1, 3), dtype=torch.bool)
    positions = build_ablation_joint_position_ids(
        base_pad,
        semantic_pad,
        suffix_pad,
        layout,
    )
    assert torch.equal(positions[:, 10:13], torch.tensor([[4, 5, 6]]))
