"""Process-local model behavior for the isolated query-conditioning ablations.

The reviewed OpenPI implementation remains untouched.  This module subclasses
it only inside the ablation entrypoints and carries the Action visibility rule
in the runtime layout, so the same rule is used by training and inference.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses

import torch

from openpi.models_pytorch import pi05_aux_queries as aux_model
import openpi.models_pytorch.preprocessing_pytorch as preprocessing

from ..integration import RoboCasaPI05AuxPolicy
from .specs import AblationSpec


@dataclasses.dataclass(frozen=True)
class AblationPrefixLayout(aux_model.PrefixLayout):
    """Prefix layout plus the query groups visible to the Action Expert."""

    action_conditioning: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.action_conditioning) - set(self.query_groups)
        if unknown:
            raise ValueError(
                f"Action-conditioning groups have no runtime span: {sorted(unknown)}"
            )


def _as_ablation_layout(
    layout: aux_model.PrefixLayout,
    action_conditioning: tuple[str, ...],
) -> AblationPrefixLayout:
    values = {
        field.name: getattr(layout, field.name)
        for field in dataclasses.fields(aux_model.PrefixLayout)
    }
    return AblationPrefixLayout(
        **values,
        action_conditioning=tuple(action_conditioning),
    )


def action_visible_prefix_mask(
    prefix_pad_mask: torch.Tensor,
    layout: aux_model.PrefixLayout,
) -> torch.Tensor:
    """Return valid prefix keys that Action is allowed to read."""

    if not isinstance(layout, AblationPrefixLayout):
        return prefix_pad_mask.to(torch.bool)
    visible = prefix_pad_mask.to(torch.bool).clone()
    for name, span in layout.query_groups.items():
        if name not in layout.action_conditioning:
            visible[:, span.start : span.end] = False
    return visible


def build_ablation_aux_train_attention(
    prefix_pad_mask: torch.Tensor,
    suffix_pad_mask: torch.Tensor,
    suffix_ar_mask: torch.Tensor,
    layout: aux_model.PrefixLayout,
) -> torch.Tensor:
    """Preserve query supervision while restricting Action-to-query edges."""

    attention = _ORIGINAL_BUILD_AUX_TRAIN_ATTENTION(
        prefix_pad_mask,
        suffix_pad_mask,
        suffix_ar_mask,
        layout,
    )
    if not isinstance(layout, AblationPrefixLayout):
        return attention
    prefix_length = prefix_pad_mask.shape[1]
    visible = action_visible_prefix_mask(prefix_pad_mask, layout)
    suffix_valid = suffix_pad_mask.to(torch.bool)
    attention[:, prefix_length:, :prefix_length] = (
        suffix_valid[:, :, None] & visible[:, None, :]
    )
    return attention


def build_ablation_joint_attention(
    paligemma_pad_mask: torch.Tensor,
    suffix_pad_mask: torch.Tensor,
    suffix_ar_mask: torch.Tensor,
    layout: aux_model.JointP2TrainLayout,
) -> torch.Tensor:
    """Keep Semantic teacher tokens hidden and apply the ablation Action mask."""

    attention = _ORIGINAL_BUILD_JOINT_ATTENTION(
        paligemma_pad_mask,
        suffix_pad_mask,
        suffix_ar_mask,
        layout,
    )
    base_layout = layout.base_layout
    if not isinstance(base_layout, AblationPrefixLayout):
        return attention
    base_pad = paligemma_pad_mask[:, : layout.semantic.start]
    visible = action_visible_prefix_mask(base_pad, base_layout)
    action = layout.action_suffix
    suffix_valid = suffix_pad_mask.to(torch.bool)
    attention[:, action.start : action.end, : layout.semantic.start] = (
        suffix_valid[:, :, None] & visible[:, None, :]
    )
    # The original joint mask already leaves Action -> Semantic empty.
    return attention


def build_ablation_joint_position_ids(
    base_prefix_pad_mask: torch.Tensor,
    semantic_pad_mask: torch.Tensor,
    suffix_pad_mask: torch.Tensor,
    layout: aux_model.JointP2TrainLayout,
) -> torch.Tensor:
    """Use the Action-visible prefix length for Action RoPE positions.

    This is identical to the reviewed implementation whenever Action reads all
    enabled queries.  For supervision-only it restores native Context->Action
    positions while the hidden query branches retain their normal positions.
    """

    positions = _ORIGINAL_BUILD_JOINT_POSITION_IDS(
        base_prefix_pad_mask,
        semantic_pad_mask,
        suffix_pad_mask,
        layout,
    )
    base_layout = layout.base_layout
    if not isinstance(base_layout, AblationPrefixLayout):
        return positions
    visible = action_visible_prefix_mask(base_prefix_pad_mask, base_layout)
    action_reference = torch.cat((visible, suffix_pad_mask.to(torch.bool)), dim=1)
    action_positions = torch.cumsum(action_reference, dim=1) - 1
    positions[:, layout.action_suffix.start : layout.action_suffix.end] = (
        action_positions[:, visible.shape[1] :]
    )
    return positions


class RoboCasaAblationPolicy(RoboCasaPI05AuxPolicy):
    """Reviewed pi0.5 auxiliary policy with one explicit ablation spec."""

    def __init__(self, config, aux_config, spec: AblationSpec) -> None:
        super().__init__(config, aux_config)
        self.ablation_spec = spec
        if not spec.geometry_enabled:
            # Semantic+Motion must not carry an unused Geometry query/head: it
            # is removed before the official base checkpoint is loaded.
            del self.geometry_queries
            del self.geometry_head
        if spec.motion_enabled != hasattr(self, "motion_queries"):
            raise ValueError(
                f"{spec.variant}: Motion module construction differs from its spec"
            )
        if spec.geometry_enabled != hasattr(self, "geometry_queries"):
            raise ValueError(
                f"{spec.variant}: Geometry module construction differs from its spec"
            )

    def expected_auxiliary_state_keys(self) -> set[str]:
        keys: set[str] = set()
        if hasattr(self, "geometry_queries"):
            keys.update(
                {
                    "geometry_queries",
                    "geometry_head.query_norm.weight",
                    "geometry_head.query_norm.bias",
                    "geometry_head.output_projection.weight",
                    "geometry_head.output_projection.bias",
                }
            )
        if hasattr(self, "motion_queries"):
            keys.update(
                {
                    "motion_queries",
                    "motion_head.query_norm.weight",
                    "motion_head.query_norm.bias",
                    "motion_head.output_projection.weight",
                    "motion_head.output_projection.bias",
                }
            )
        return keys

    def _append_aux_queries(
        self,
        context_embeddings: torch.Tensor,
        context_pad_mask: torch.Tensor,
        *,
        view_spans: Mapping[str, aux_model.TokenSpan],
        real_view_names: tuple[str, ...],
        padded_view_names: tuple[str, ...],
        language_span: aux_model.TokenSpan,
    ) -> tuple[torch.Tensor, torch.Tensor, AblationPrefixLayout]:
        embeddings, pads, layout = super()._append_aux_queries(
            context_embeddings,
            context_pad_mask,
            view_spans=view_spans,
            real_view_names=real_view_names,
            padded_view_names=padded_view_names,
            language_span=language_span,
        )
        return (
            embeddings,
            pads,
            _as_ablation_layout(layout, self.ablation_spec.action_conditioning),
        )

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        """Use the same Action-query visibility at checkpoint evaluation time."""

        batch = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise(
                (batch, self.config.action_horizon, self.config.action_dim),
                device,
            )
        processed = preprocessing.preprocess_observation_pytorch(
            observation,
            train=False,
        )
        (
            context,
            context_pad,
            view_spans,
            real_views,
            padded_views,
            language_span,
        ) = self._embed_context_with_layout(
            processed.images,
            processed.image_masks,
            processed.tokenized_prompt,
            processed.tokenized_prompt_mask,
        )
        prefix, prefix_pad, layout = self._append_aux_queries(
            context,
            context_pad,
            view_spans=view_spans,
            real_view_names=real_views,
            padded_view_names=padded_views,
            language_span=language_span,
        )
        prefix_attention = aux_model.build_explicit_aux_prefix_attention(
            prefix_pad,
            layout,
        )
        prefix_position_ids = torch.cumsum(prefix_pad, dim=1) - 1
        language_model = self.paligemma_with_expert.paligemma.language_model
        language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(prefix_attention),
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )

        action_prefix_pad = action_visible_prefix_mask(prefix_pad, layout)
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        actions = noise
        current_time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while current_time >= -dt / 2:
            actions = actions + dt * self.denoise_step(
                processed.state,
                action_prefix_pad,
                past_key_values,
                actions,
                current_time.expand(batch),
            )
            current_time += dt
        return actions


# Saved before process-local installation so non-ablation layouts retain the
# exact reviewed implementation and recursion is impossible.
_ORIGINAL_BUILD_AUX_TRAIN_ATTENTION = aux_model.build_explicit_aux_train_attention
_ORIGINAL_BUILD_JOINT_ATTENTION = aux_model.build_joint_p2_attention
_ORIGINAL_BUILD_JOINT_POSITION_IDS = aux_model.build_joint_p2_position_ids
