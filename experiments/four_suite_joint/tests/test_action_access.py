from __future__ import annotations

from four_suite_experiments.action_access import install_action_access_policy
from four_suite_experiments.action_access import mask_action_rows
import torch

from openpi.models_pytorch import pi05_aux_queries as aux
from openpi.models_pytorch.pi05_aux_queries import JointP2TrainLayout
from openpi.models_pytorch.pi05_aux_queries import PrefixLayout
from openpi.models_pytorch.pi05_aux_queries import TokenSpan


def _layout() -> PrefixLayout:
    return PrefixLayout(
        view_spans={"base_0_rgb": TokenSpan(0, 2)},
        real_view_names=("base_0_rgb",),
        padded_view_names=(),
        language=TokenSpan(2, 4),
        context=TokenSpan(0, 4),
        geometry=TokenSpan(4, 6),
        motion=TokenSpan(6, 8),
        ground=None,
        action_suffix=TokenSpan(8, 11),
    )


def test_block_both_changes_only_action_to_query_edges() -> None:
    original = torch.ones((2, 11, 11), dtype=torch.bool)
    masked = mask_action_rows(
        original,
        action_start=8,
        layout=_layout(),
        blocked_groups={"geometry", "motion"},
    )
    assert not masked[:, 8:, 4:8].any()
    assert masked[:, :8, :].all()
    assert masked[:, 8:, :4].all()
    assert masked[:, 8:, 8:].all()
    assert original.all(), "input mask must not be mutated"


def test_block_geometry_retains_motion_access() -> None:
    masked = mask_action_rows(
        torch.ones((1, 11, 11), dtype=torch.bool),
        action_start=8,
        layout=_layout(),
        blocked_groups={"geometry"},
    )
    assert not masked[:, 8:, 4:6].any()
    assert masked[:, 8:, 6:8].all()


def test_installed_policy_masks_both_training_attention_paths() -> None:
    install_action_access_policy({"geometry", "motion"})
    layout = _layout()
    prefix_pad = torch.ones((1, 8), dtype=torch.bool)
    suffix_pad = torch.ones((1, 3), dtype=torch.bool)
    suffix_ar = torch.zeros((1, 3), dtype=torch.bool)
    explicit = aux.build_explicit_aux_train_attention(
        prefix_pad, suffix_pad, suffix_ar, layout
    )
    assert explicit[:, 8:, :4].all()
    assert not explicit[:, 8:, 4:8].any()
    assert explicit[:, 4:8, :8].any(), "query computation must remain intact"

    joint_layout = JointP2TrainLayout(
        base_layout=layout,
        semantic=TokenSpan(8, 10),
        action_suffix=TokenSpan(10, 13),
    )
    joint = aux.build_joint_p2_attention(
        torch.ones((1, 10), dtype=torch.bool),
        suffix_pad,
        suffix_ar,
        joint_layout,
    )
    assert joint[:, 10:, :4].all()
    assert not joint[:, 10:, 4:8].any()
    assert not joint[:, 10:, 8:10].any(), "Semantic teacher tokens remain isolated"
