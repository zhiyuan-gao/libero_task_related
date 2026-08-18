"""Optional P1/P2 dedicated-query extensions for the official PyTorch pi0.5 policy.

The disabled mode delegates to :class:`PI0Pytorch` unchanged. Enabled modes add
explicitly isolated VLM-prefix query groups that are visible to the action expert.
P2 semantic supervision uses a separate native VLM autoregressive language pass.
Teacher-forced semantic tokens are never inserted into the action forward. P2 is
a strict prefix extension of P1: Geometry occupies the first eight auxiliary
positions, followed by eight Grounding queries.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import math
from typing import Literal

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.auxiliary_heads import MeanQueryProjectionHead
from openpi.models_pytorch.auxiliary_heads import QueryConditionedPatchMaskHead
from openpi.models_pytorch.auxiliary_heads import grounding_focal_dice_loss
from openpi.models_pytorch.auxiliary_heads import masked_standardized_mse
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.policy_aux_preprocessing import patch_foreground_coverage
from openpi.models_pytorch.policy_aux_preprocessing import preprocess_observation_and_ground_masks_pytorch
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

PolicyAuxMode = Literal["none", "geometry", "ground_geometry_semantic_lm"]

# New branches use independent fixed RNG streams so shared P1/P2 Geometry
# parameters do not depend on model mode or module-construction order.
GEOMETRY_QUERY_INIT_SEED = 2026081801
GEOMETRY_HEAD_INIT_SEED = 2026081802
GROUND_QUERY_INIT_SEED = 2026081811
GROUND_HEAD_INIT_SEED = 2026081812


@dataclasses.dataclass(frozen=True)
class PolicyAuxConfig:
    mode: PolicyAuxMode = "none"
    num_ground_queries: int = 8
    num_geometry_queries: int = 8
    geometry_target_dim: int = 2048
    ground_mask_dim: int = 256
    ground_focal_alpha: float = 0.25
    ground_focal_gamma: float = 2.0
    lambda_sem: float | None = None
    lambda_ground: float | None = None
    lambda_geo: float | None = None
    semantic_annotation_root: str | None = None
    ground_mask_root: str | None = None
    geometry_cache_root: str | None = None
    geometry_normalization_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("none", "geometry", "ground_geometry_semantic_lm"):
            raise ValueError(f"Unsupported policy_aux_mode: {self.mode}")
        if self.num_ground_queries != 8 or self.num_geometry_queries != 8:
            raise ValueError("P1/P2 v0 require eight Grounding/Geometry queries")
        if self.geometry_target_dim != 2048:
            raise ValueError("P1/P2 v0 Geometry target dimension is frozen at 2048")
        for name in ("lambda_sem", "lambda_ground", "lambda_geo"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclasses.dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid token span [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclasses.dataclass(frozen=True)
class PrefixLayout:
    view_spans: Mapping[str, TokenSpan]
    real_view_names: tuple[str, ...]
    padded_view_names: tuple[str, ...]
    language: TokenSpan
    context: TokenSpan
    ground: TokenSpan | None
    geometry: TokenSpan | None
    action_suffix: TokenSpan | None = None

    @property
    def query_groups(self) -> dict[str, TokenSpan]:
        return {
            name: span
            for name, span in (
                ("geometry", self.geometry),
                ("ground", self.ground),
            )
            if span is not None
        }


@dataclasses.dataclass
class PolicyAuxTargets:
    geometry: torch.Tensor | None = None
    geometry_valid: torch.Tensor | None = None
    geometry_mean: torch.Tensor | None = None
    geometry_std: torch.Tensor | None = None
    ground_masks: Mapping[str, torch.Tensor] | None = None
    ground_valid_views: torch.Tensor | None = None
    semantic_input_ids: torch.Tensor | None = None
    semantic_labels: torch.Tensor | None = None
    semantic_loss_mask: torch.Tensor | None = None


def build_explicit_aux_prefix_attention(
    pad_mask: torch.Tensor,
    layout: PrefixLayout,
) -> torch.Tensor:
    """Build the frozen P1/P2 prefix connectivity as an explicit 2-D mask."""

    if pad_mask.ndim != 2:
        raise ValueError("pad_mask must have shape [B,N]")
    batch, length = pad_mask.shape
    if layout.context.start != 0 or layout.context.end > length:
        raise ValueError("Context span is inconsistent with the prefix length")
    for name, span in layout.query_groups.items():
        if span.start < layout.context.end or span.end > length:
            raise ValueError(f"Invalid {name} query span")

    connectivity = torch.zeros((batch, length, length), dtype=torch.bool, device=pad_mask.device)
    context = layout.context
    connectivity[:, context.start : context.end, context.start : context.end] = True
    for span in layout.query_groups.values():
        connectivity[:, span.start : span.end, context.start : context.end] = True
        connectivity[:, span.start : span.end, span.start : span.end] = True

    valid = pad_mask.to(torch.bool)
    return connectivity & valid[:, :, None] & valid[:, None, :]


def build_explicit_aux_train_attention(
    prefix_pad_mask: torch.Tensor,
    suffix_pad_mask: torch.Tensor,
    suffix_ar_mask: torch.Tensor,
    layout: PrefixLayout,
) -> torch.Tensor:
    """Combine explicit prefix isolation with the official action-suffix mask."""

    prefix_attention = build_explicit_aux_prefix_attention(prefix_pad_mask, layout)
    suffix_attention = make_att_2d_masks(suffix_pad_mask, suffix_ar_mask)
    batch, prefix_length = prefix_pad_mask.shape
    suffix_length = suffix_pad_mask.shape[1]
    full = torch.zeros(
        (batch, prefix_length + suffix_length, prefix_length + suffix_length),
        dtype=torch.bool,
        device=prefix_pad_mask.device,
    )
    full[:, :prefix_length, :prefix_length] = prefix_attention
    full[:, prefix_length:, :prefix_length] = (
        suffix_pad_mask.to(torch.bool)[:, :, None] & prefix_pad_mask.to(torch.bool)[:, None, :]
    )
    full[:, prefix_length:, prefix_length:] = suffix_attention
    return full


def build_native_semantic_lm_attention(
    context_pad_mask: torch.Tensor,
    teacher_input_mask: torch.Tensor,
) -> torch.Tensor:
    """Build prefix-LM attention for the separate native semantic LM pass.

    Valid image/instruction context tokens interact bidirectionally and cannot
    read teacher-forced tokens.  Each valid teacher token reads all valid
    context and its causal teacher prefix.  This mask is never used by the
    action forward.
    """

    if context_pad_mask.ndim != 2 or teacher_input_mask.ndim != 2:
        raise ValueError("Semantic context and teacher masks must be rank-2")
    if context_pad_mask.shape[0] != teacher_input_mask.shape[0]:
        raise ValueError("Semantic context and teacher batch sizes differ")
    context_valid = context_pad_mask.to(torch.bool)
    teacher_valid = teacher_input_mask.to(torch.bool)
    batch, context_length = context_valid.shape
    teacher_length = teacher_valid.shape[1]
    total_length = context_length + teacher_length
    attention = torch.zeros(
        (batch, total_length, total_length),
        dtype=torch.bool,
        device=context_pad_mask.device,
    )
    attention[:, :context_length, :context_length] = context_valid[:, :, None] & context_valid[:, None, :]
    attention[:, context_length:, :context_length] = teacher_valid[:, :, None] & context_valid[:, None, :]
    causal = torch.tril(
        torch.ones(
            (teacher_length, teacher_length),
            dtype=torch.bool,
            device=context_pad_mask.device,
        )
    )
    attention[:, context_length:, context_length:] = (
        causal[None] & teacher_valid[:, :, None] & teacher_valid[:, None, :]
    )
    return attention


class PI05AuxPolicy(PI0Pytorch):
    """Official pi0.5 action policy with optional P1/P2 query groups."""

    def __init__(self, config, aux_config: PolicyAuxConfig) -> None:
        super().__init__(config)
        if config.pi05 is not True or config.action_horizon != 10 or config.discrete_state_input is not False:
            raise ValueError("P1/P2 require pi05=True, action_horizon=10, discrete_state_input=False")
        self.aux_config = aux_config
        self.hidden_dim = int(self.paligemma_with_expert.paligemma.language_model.config.hidden_size)

        if aux_config.mode == "geometry":
            self.geometry_queries = self._new_queries(aux_config.num_geometry_queries, seed=GEOMETRY_QUERY_INIT_SEED)
            self.geometry_head = self._new_seeded_module(
                GEOMETRY_HEAD_INIT_SEED,
                lambda: MeanQueryProjectionHead(self.hidden_dim, aux_config.geometry_target_dim),
            )
        elif aux_config.mode == "ground_geometry_semantic_lm":
            self.geometry_queries = self._new_queries(aux_config.num_geometry_queries, seed=GEOMETRY_QUERY_INIT_SEED)
            self.geometry_head = self._new_seeded_module(
                GEOMETRY_HEAD_INIT_SEED,
                lambda: MeanQueryProjectionHead(self.hidden_dim, aux_config.geometry_target_dim),
            )
            self.ground_queries = self._new_queries(aux_config.num_ground_queries, seed=GROUND_QUERY_INIT_SEED)
            self.ground_head = self._new_seeded_module(
                GROUND_HEAD_INIT_SEED,
                lambda: QueryConditionedPatchMaskHead(self.hidden_dim, aux_config.ground_mask_dim),
            )

    def _new_queries(self, count: int, *, seed: int) -> nn.Parameter:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            parameter = nn.Parameter(torch.empty(count, self.hidden_dim))
            nn.init.trunc_normal_(parameter, mean=0.0, std=0.02)
        return parameter

    @staticmethod
    def _new_seeded_module(seed: int, constructor):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return constructor()

    @property
    def aux_enabled(self) -> bool:
        return self.aux_config.mode != "none"

    def expected_auxiliary_state_keys(self) -> set[str]:
        if self.aux_config.mode == "none":
            return set()
        keys = {
            "geometry_queries",
            "geometry_head.query_norm.weight",
            "geometry_head.query_norm.bias",
            "geometry_head.output_projection.weight",
            "geometry_head.output_projection.bias",
        }
        if self.aux_config.mode == "ground_geometry_semantic_lm":
            keys.update(
                {
                    "ground_queries",
                    "ground_head.logit_bias",
                    "ground_head.query_norm.weight",
                    "ground_head.query_norm.bias",
                    "ground_head.patch_norm.weight",
                    "ground_head.patch_norm.bias",
                    "ground_head.query_projection.weight",
                    "ground_head.query_projection.bias",
                    "ground_head.patch_projection.weight",
                    "ground_head.patch_projection.bias",
                }
            )
        return keys

    def load_official_base_checkpoint(self, checkpoint_path: str, *, device: str = "cpu") -> dict[str, list[str]]:
        """Load official base weights and allow only the exact new P1/P2 keys to be missing."""

        missing, unexpected = safetensors.torch.load_model(self, checkpoint_path, strict=False, device=device)
        missing_set = set(missing)
        expected_missing = self.expected_auxiliary_state_keys()
        if missing_set != expected_missing or unexpected:
            raise RuntimeError(
                "Official base checkpoint mismatch: "
                f"expected_missing={sorted(expected_missing)}, missing={sorted(missing_set)}, "
                f"unexpected={sorted(unexpected)}"
            )
        return {"missing": sorted(missing), "unexpected": sorted(unexpected)}

    def _append_aux_queries(
        self,
        context_embeddings: torch.Tensor,
        context_pad_mask: torch.Tensor,
        *,
        view_spans: Mapping[str, TokenSpan],
        real_view_names: tuple[str, ...],
        padded_view_names: tuple[str, ...],
        language_span: TokenSpan,
    ) -> tuple[torch.Tensor, torch.Tensor, PrefixLayout]:
        batch, context_length, _ = context_embeddings.shape
        embeddings = [context_embeddings]
        pads = [context_pad_mask.to(torch.bool)]
        cursor = context_length
        spans: dict[str, TokenSpan | None] = {"geometry": None, "ground": None}

        for name in ("geometry", "ground"):
            parameter = getattr(self, f"{name}_queries", None)
            if parameter is None:
                continue
            count = int(parameter.shape[0])
            spans[name] = TokenSpan(cursor, cursor + count)
            embeddings.append(parameter[None].expand(batch, -1, -1).to(context_embeddings.dtype))
            pads.append(torch.ones((batch, count), dtype=torch.bool, device=context_embeddings.device))
            cursor += count

        layout = PrefixLayout(
            view_spans=dict(view_spans),
            real_view_names=real_view_names,
            padded_view_names=padded_view_names,
            language=language_span,
            context=TokenSpan(0, context_length),
            ground=spans["ground"],
            geometry=spans["geometry"],
        )
        return torch.cat(embeddings, dim=1), torch.cat(pads, dim=1), layout

    def _embed_context_with_layout(
        self,
        images: Mapping[str, torch.Tensor],
        image_masks: Mapping[str, torch.Tensor],
        language_tokens: torch.Tensor,
        language_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, TokenSpan], tuple[str, ...], tuple[str, ...], TokenSpan]:
        """Mirror official ``embed_prefix`` while recording spans from runtime shapes."""

        embeddings = []
        pads = []
        view_spans: dict[str, TokenSpan] = {}
        real_views = []
        padded_views = []
        cursor = 0

        for name, image in images.items():
            image_embedding = self._apply_checkpoint(self.paligemma_with_expert.embed_image, image)
            batch, token_count = image_embedding.shape[:2]
            view_spans[name] = TokenSpan(cursor, cursor + token_count)
            cursor += token_count
            view_valid = image_masks[name].to(torch.bool)
            embeddings.append(image_embedding)
            pads.append(view_valid[:, None].expand(batch, token_count))
            if bool(view_valid.all()):
                real_views.append(name)
            elif bool((~view_valid).all()):
                padded_views.append(name)
            else:
                raise ValueError(f"View {name} mixes real/padded slots within one batch")

        language_embedding = self._apply_checkpoint(self.paligemma_with_expert.embed_language_tokens, language_tokens)
        language_embedding = language_embedding * self.hidden_dim**0.5
        language_span = TokenSpan(cursor, cursor + language_embedding.shape[1])
        embeddings.append(language_embedding)
        pads.append(language_masks.to(torch.bool))
        return (
            torch.cat(embeddings, dim=1),
            torch.cat(pads, dim=1),
            view_spans,
            tuple(real_views),
            tuple(padded_views),
            language_span,
        )

    def _native_semantic_lm_decode(
        self,
        context_embeddings: torch.Tensor,
        context_pad_mask: torch.Tensor,
        language_span: TokenSpan,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Native autoregressive semantic text objective from image+instruction context."""

        if labels.shape != loss_mask.shape:
            raise ValueError("Semantic label/loss-mask shapes differ")
        if input_ids.ndim != 2 or labels.ndim != 2 or input_ids.shape[0] != labels.shape[0]:
            raise ValueError("Semantic teacher tensors must be rank-2 with equal batch size")
        if input_ids.shape[1] + 1 != labels.shape[1]:
            raise ValueError("Semantic inputs must contain labels shifted left by one token")
        if context_embeddings.ndim != 3 or context_pad_mask.shape != context_embeddings.shape[:2]:
            raise ValueError("Semantic context embedding/pad shapes differ")
        if language_span.end > context_embeddings.shape[1] or language_span.length == 0:
            raise ValueError("Semantic language span is outside the context")

        batch, _ = labels.shape
        language_valid = context_pad_mask[:, language_span.start : language_span.end].to(torch.bool)
        if not bool(language_valid.any(dim=1).all()):
            raise ValueError("Every semantic sample requires at least one valid instruction token")
        # Official prompt padding is right-aligned after the valid prompt prefix.
        seen_padding = (~language_valid).to(torch.int64).cumsum(dim=1) > 0
        if bool((language_valid & seen_padding).any()):
            raise ValueError("Semantic instruction masks must be a contiguous valid prefix")
        relative_indices = torch.arange(language_span.length, device=input_ids.device)[None]
        last_language_offset = torch.where(
            language_valid, relative_indices, torch.full_like(relative_indices, -1)
        ).amax(dim=1)
        anchor_indices = language_span.start + last_language_offset

        token_embeddings = self._apply_checkpoint(self.paligemma_with_expert.embed_language_tokens, input_ids)
        token_embeddings = token_embeddings * self.hidden_dim**0.5
        teacher_input_mask = loss_mask[:, 1:].to(torch.bool)
        decoder_embeddings = torch.cat((context_embeddings, token_embeddings), dim=1)
        attention = build_native_semantic_lm_attention(context_pad_mask, teacher_input_mask)
        valid = torch.cat((context_pad_mask.to(torch.bool), teacher_input_mask), dim=1)
        position_ids = (torch.cumsum(valid, dim=1) - 1).clamp_min(0)
        language_model = self.paligemma_with_expert.paligemma.language_model
        language_model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs = language_model.forward(
            inputs_embeds=decoder_embeddings,
            attention_mask=self._prepare_attention_masks_4d(attention),
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            adarms_cond=None,
        ).last_hidden_state
        batch_indices = torch.arange(batch, device=input_ids.device)
        prediction_states = torch.cat(
            (
                outputs[batch_indices, anchor_indices][:, None],
                outputs[:, context_embeddings.shape[1] :],
            ),
            dim=1,
        )
        lm_head = self.paligemma_with_expert.paligemma.lm_head
        logits = lm_head(prediction_states.to(lm_head.weight.dtype)).float()
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        mask = loss_mask.to(torch.bool)
        if not bool(mask.any()):
            loss = logits.sum() * 0.0
            accuracy = logits.sum() * 0.0
            exact_match = logits.sum() * 0.0
        else:
            loss = per_token[mask].mean()
            correct = logits.argmax(dim=-1) == labels
            accuracy = correct[mask].float().mean()
            exact_match = (correct | ~mask).all(dim=1).float().mean()
        return {
            "loss": loss,
            "token_accuracy": accuracy,
            "teacher_forced_exact_match": exact_match,
            "logits": logits,
        }

    def forward_with_aux(
        self,
        observation,
        actions: torch.Tensor,
        aux_targets: PolicyAuxTargets,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | PrefixLayout | dict[str, torch.Tensor]]:
        if not self.aux_enabled:
            raise RuntimeError("forward_with_aux requires an enabled auxiliary mode")

        if self.aux_config.mode == "ground_geometry_semantic_lm":
            if aux_targets.ground_masks is None:
                raise ValueError("P2 requires policy-canvas Grounding masks")
            synchronized = preprocess_observation_and_ground_masks_pytorch(
                observation,
                aux_targets.ground_masks,
                train=True,
            )
            processed = synchronized.observation
            transformed_ground_masks = synchronized.ground_masks
        else:
            processed = _preprocessing.preprocess_observation_pytorch(observation, train=True)
            transformed_ground_masks = None
        images = processed.images
        image_masks = processed.image_masks
        state = processed.state
        language_tokens = processed.tokenized_prompt
        language_masks = processed.tokenized_prompt_mask
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        action_target = noise - actions

        context, context_pad, view_spans, real_views, padded_views, language_span = self._embed_context_with_layout(
            images, image_masks, language_tokens, language_masks
        )
        prefix, prefix_pad, layout = self._append_aux_queries(
            context,
            context_pad,
            view_spans=view_spans,
            real_view_names=real_views,
            padded_view_names=padded_views,
            language_span=language_span,
        )
        suffix, suffix_pad, suffix_ar, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix = prefix.to(torch.bfloat16)
            suffix = suffix.to(torch.bfloat16)

        attention = build_explicit_aux_train_attention(prefix_pad, suffix_pad, suffix_ar, layout)
        all_pad = torch.cat((prefix_pad, suffix_pad), dim=1)
        position_ids = torch.cumsum(all_pad, dim=1) - 1
        attention_4d = self._prepare_attention_masks_4d(attention)

        def joint_forward(prefix, suffix, attention_4d, position_ids, adarms_cond):
            return self.paligemma_with_expert.forward(
                attention_mask=attention_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix, suffix],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )[0]

        prefix_output, suffix_output = self._apply_checkpoint(
            joint_forward, prefix, suffix, attention_4d, position_ids, adarms_cond
        )
        action_velocity = self.action_out_proj(suffix_output[:, -self.config.action_horizon :].float())
        action_loss_per_element = F.mse_loss(action_velocity, action_target, reduction="none")
        losses: dict[str, torch.Tensor] = {"action": action_loss_per_element.mean()}
        diagnostics: dict[str, torch.Tensor] = {}

        if layout.geometry is not None:
            geometry_states = prefix_output[:, layout.geometry.start : layout.geometry.end]
            geometry_prediction = self.geometry_head(geometry_states)
            required = (
                aux_targets.geometry,
                aux_targets.geometry_valid,
                aux_targets.geometry_mean,
                aux_targets.geometry_std,
            )
            if any(value is None for value in required):
                raise ValueError("Enabled Geometry branch requires target, valid, mean, and std")
            losses["geometry"] = masked_standardized_mse(
                geometry_prediction,
                aux_targets.geometry,
                aux_targets.geometry_valid,
                aux_targets.geometry_mean,
                aux_targets.geometry_std,
            )
            diagnostics["geometry_prediction"] = geometry_prediction
            geometry_valid = aux_targets.geometry_valid.to(torch.bool)
            if bool(geometry_valid.any()):
                diagnostics["geometry_raw_cosine"] = F.cosine_similarity(
                    geometry_prediction[geometry_valid].float(),
                    aux_targets.geometry[geometry_valid].float(),
                    dim=-1,
                ).mean()
                diagnostics["geometry_prediction_norm"] = (
                    geometry_prediction[geometry_valid].float().norm(dim=-1).mean()
                )
                diagnostics["geometry_prediction_variance"] = (
                    geometry_prediction[geometry_valid].float().var(dim=0, unbiased=False).mean()
                )

        if layout.ground is not None:
            if transformed_ground_masks is None or aux_targets.ground_valid_views is None:
                raise ValueError("Enabled Grounding branch requires masks and valid views")
            patch_states = torch.stack(
                [
                    prefix_output[:, layout.view_spans[name].start : layout.view_spans[name].end]
                    for name in layout.real_view_names
                ],
                dim=1,
            )
            patch_targets = []
            for name in layout.real_view_names:
                if name not in transformed_ground_masks:
                    raise ValueError(f"Missing transformed Grounding mask for real view {name}")
                patch_count = layout.view_spans[name].length
                grid_size = math.isqrt(patch_count)
                if grid_size * grid_size != patch_count:
                    raise ValueError(f"Image token count for {name} is not a square patch grid: {patch_count}")
                patch_targets.append(
                    patch_foreground_coverage(
                        transformed_ground_masks[name],
                        grid_height=grid_size,
                        grid_width=grid_size,
                    )
                )
            ground_patch_coverage = torch.stack(patch_targets, dim=1)
            if aux_targets.ground_valid_views.shape != ground_patch_coverage.shape[:2]:
                raise ValueError("Ground target valid-view shape does not match runtime real views")
            ground_states = prefix_output[:, layout.ground.start : layout.ground.end]
            ground_logits = self.ground_head(ground_states, patch_states)
            ground = grounding_focal_dice_loss(
                ground_logits,
                ground_patch_coverage,
                aux_targets.ground_valid_views,
                alpha=self.aux_config.ground_focal_alpha,
                gamma=self.aux_config.ground_focal_gamma,
            )
            losses["ground"] = ground["loss"]
            diagnostics.update({f"ground_{key}": value for key, value in ground.items() if key != "loss"})
            for view_index, view_name in enumerate(layout.real_view_names):
                diagnostic_name = "agent" if view_name == "base_0_rgb" else "wrist"
                for metric in (
                    "focal_loss",
                    "dice_loss",
                    "dice_score",
                    "precision",
                    "recall",
                    "iou",
                ):
                    diagnostics[f"ground_{diagnostic_name}_{metric}"] = ground[f"{metric}_by_view"][view_index]
                diagnostics[f"ground_{diagnostic_name}_valid_count"] = ground["valid_count_by_view"][view_index]
            diagnostics["ground_logits"] = ground_logits
            diagnostics["ground_patch_coverage"] = ground_patch_coverage

        if self.aux_config.mode == "ground_geometry_semantic_lm":
            required_semantic = (
                aux_targets.semantic_input_ids,
                aux_targets.semantic_labels,
                aux_targets.semantic_loss_mask,
            )
            if any(value is None for value in required_semantic):
                raise ValueError("Enabled Semantic branch requires teacher-forcing tensors")
            semantic = self._native_semantic_lm_decode(
                context,
                context_pad,
                language_span,
                aux_targets.semantic_input_ids,
                aux_targets.semantic_labels,
                aux_targets.semantic_loss_mask,
            )
            losses["semantic"] = semantic["loss"]
            diagnostics["semantic_token_accuracy"] = semantic["token_accuracy"]
            diagnostics["semantic_teacher_forced_exact_match"] = semantic["teacher_forced_exact_match"]
            diagnostics["semantic_supervised_token_count"] = aux_targets.semantic_loss_mask.to(torch.int64).sum()

        required_lambdas = {
            "geometry": self.aux_config.lambda_geo,
            "ground": self.aux_config.lambda_ground,
            "semantic": self.aux_config.lambda_sem,
        }
        missing_lambdas = [name for name in losses if name in required_lambdas and required_lambdas[name] is None]
        if missing_lambdas:
            raise RuntimeError(
                f"Auxiliary loss coefficients require calibration/human freeze before training: {missing_lambdas}"
            )
        total = losses["action"]
        for name, coefficient in required_lambdas.items():
            if name in losses:
                weighted = float(coefficient) * losses[name]
                diagnostics[f"weighted_{name}_contribution"] = weighted.detach()
                total = total + weighted
        losses["total"] = total
        return {
            "losses": losses,
            "diagnostics": diagnostics,
            "layout": dataclasses.replace(
                layout,
                action_suffix=TokenSpan(prefix.shape[1], prefix.shape[1] + suffix.shape[1]),
            ),
            "action_loss_per_element": action_loss_per_element,
        }

    def forward(self, observation, actions, aux_targets=None, noise=None, time=None):
        if not self.aux_enabled:
            if aux_targets is not None:
                raise ValueError("policy_aux_mode=none must not receive auxiliary targets")
            return super().forward(observation, actions, noise=noise, time=time)
        if aux_targets is None:
            raise ValueError("Enabled auxiliary mode requires PolicyAuxTargets during training")
        return self.forward_with_aux(observation, actions, aux_targets, noise=noise, time=time)

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10):
        if not self.aux_enabled:
            return super().sample_actions(device, observation, noise=noise, num_steps=num_steps)

        batch = observation.state.shape[0]
        if noise is None:
            noise = self.sample_noise((batch, self.config.action_horizon, self.config.action_dim), device)
        processed = _preprocessing.preprocess_observation_pytorch(observation, train=False)
        context, context_pad, view_spans, real_views, padded_views, language_span = self._embed_context_with_layout(
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
        prefix_attention = build_explicit_aux_prefix_attention(prefix_pad, layout)
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

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        actions = noise
        current_time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while current_time >= -dt / 2:
            actions = actions + dt * self.denoise_step(
                processed.state,
                prefix_pad,
                past_key_values,
                actions,
                current_time.expand(batch),
            )
            current_time += dt
        return actions
