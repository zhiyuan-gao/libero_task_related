"""Attention-only controls for what the Action Expert may read."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from openpi.models_pytorch import pi05_aux_queries as aux
from openpi.models_pytorch import preprocessing_pytorch

VALID_GROUPS = frozenset({"geometry", "motion"})
_ORIGINAL_EXPLICIT = aux.build_explicit_aux_train_attention
_ORIGINAL_JOINT = aux.build_joint_p2_attention
_ORIGINAL_SAMPLE_ACTIONS = aux.PI05AuxPolicy.sample_actions


def _normalized_groups(groups: Iterable[str]) -> frozenset[str]:
    result = frozenset(str(group) for group in groups)
    unknown = result - VALID_GROUPS
    if unknown:
        raise ValueError(f"unknown Action input groups: {sorted(unknown)}")
    return result


def mask_action_rows(
    attention: torch.Tensor,
    *,
    action_start: int,
    layout,
    blocked_groups: Iterable[str],
) -> torch.Tensor:
    """Return a mask copy with only Action-to-selected-query edges removed."""

    blocked = _normalized_groups(blocked_groups)
    result = attention.clone()
    for name in blocked:
        span = getattr(layout, name, None)
        if span is not None:
            result[:, action_start:, span.start : span.end] = False
    return result


def install_action_access_policy(blocked_groups: Iterable[str]) -> None:
    """Install one process-wide policy before model construction or data loading."""

    blocked = _normalized_groups(blocked_groups)
    already = getattr(aux, "_four_suite_blocked_action_groups", None)
    if already is not None:
        if already != blocked:
            raise RuntimeError(
                f"Action access policy already installed as {sorted(already)}"
            )
        return
    aux._four_suite_blocked_action_groups = blocked  # noqa: SLF001
    if not blocked:
        return

    def explicit(prefix_pad_mask, suffix_pad_mask, suffix_ar_mask, layout):
        full = _ORIGINAL_EXPLICIT(
            prefix_pad_mask, suffix_pad_mask, suffix_ar_mask, layout
        )
        return mask_action_rows(
            full,
            action_start=prefix_pad_mask.shape[1],
            layout=layout,
            blocked_groups=blocked,
        )

    def joint(paligemma_pad_mask, suffix_pad_mask, suffix_ar_mask, layout):
        full = _ORIGINAL_JOINT(
            paligemma_pad_mask, suffix_pad_mask, suffix_ar_mask, layout
        )
        return mask_action_rows(
            full,
            action_start=layout.action_suffix.start,
            layout=layout.base_layout,
            blocked_groups=blocked,
        )

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        if not self.aux_enabled:
            return _ORIGINAL_SAMPLE_ACTIONS(
                self, device, observation, noise=noise, num_steps=num_steps
            )
        batch = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise(
                (batch, self.config.action_horizon, self.config.action_dim), device
            )
        processed = preprocessing_pytorch.preprocess_observation_pytorch(
            observation, train=False
        )
        context, context_pad, view_spans, real_views, padded_views, language_span = (
            self._embed_context_with_layout(
                processed.images,
                processed.image_masks,
                processed.tokenized_prompt,
                processed.tokenized_prompt_mask,
            )
        )
        prefix, prefix_pad, layout = self._append_aux_queries(
            context,
            context_pad,
            view_spans=view_spans,
            real_view_names=real_views,
            padded_view_names=padded_views,
            language_span=language_span,
        )
        prefix_attention = aux.build_explicit_aux_prefix_attention(prefix_pad, layout)
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

        action_prefix_mask = prefix_pad.clone().to(torch.bool)
        for name in blocked:
            span = getattr(layout, name, None)
            if span is not None:
                action_prefix_mask[:, span.start : span.end] = False
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        actions = noise
        current_time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while current_time >= -dt / 2:
            actions = actions + dt * _denoise_step_with_action_mask(
                self,
                processed.state,
                prefix_pad,
                action_prefix_mask,
                past_key_values,
                actions,
                current_time.expand(batch),
            )
            current_time += dt
        return actions

    aux.build_explicit_aux_train_attention = explicit
    aux.build_joint_p2_attention = joint
    aux.PI05AuxPolicy.sample_actions = sample_actions


def _denoise_step_with_action_mask(
    self,
    state,
    prefix_pad_mask,
    action_prefix_mask,
    past_key_values,
    x_t,
    timestep,
):
    """Existing denoise step with distinct attention and RoPE prefix masks."""

    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
        state, x_t, timestep
    )
    suffix_len = suffix_pad_masks.shape[1]
    batch_size, prefix_len = prefix_pad_mask.shape
    if action_prefix_mask.shape != prefix_pad_mask.shape:
        raise ValueError("Action prefix mask shape differs from cached prefix")
    prefix_attention = action_prefix_mask[:, None, :].expand(
        batch_size, suffix_len, prefix_len
    )
    suffix_attention = aux.make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_attention = torch.cat([prefix_attention, suffix_attention], dim=2)

    # Position offsets must use the complete cached prefix, even when some keys are inaccessible.
    prefix_offsets = torch.sum(prefix_pad_mask, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    attention_4d = self._prepare_attention_masks_4d(full_attention)
    self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
    outputs_embeds, _ = self.paligemma_with_expert.forward(
        attention_mask=attention_4d,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(
        dtype=torch.float32
    )
    return self.action_out_proj(suffix_out)
