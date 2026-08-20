"""Optional P1/P2 dedicated-query extensions for the official PyTorch pi0.5 policy.

The disabled mode delegates to :class:`PI0Pytorch` unchanged. Enabled modes add
explicitly isolated VLM-prefix query groups that are visible to the action expert.
P2 semantic supervision retains the native VLM autoregressive language objective.
Teacher-forced semantic tokens share the production training-time PaliGemma
transformer forward but are explicitly masked out of the action/Geometry/Ground
paths. P2 is a strict prefix extension of P1: Geometry occupies the first eight
auxiliary positions, followed by eight Grounding queries. Inference remains
teacher-free.
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
from openpi.models_pytorch.auxiliary_heads import masked_standardized_smooth_l1
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.policy_aux_preprocessing import patch_foreground_coverage
from openpi.models_pytorch.policy_aux_preprocessing import preprocess_observation_and_ground_masks_pytorch
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

PolicyAuxMode = Literal[
    "none", "geometry", "semantic_geometry", "semantic_geometry_motion", "ground_geometry_semantic_lm"
]
SemanticImplementation = Literal["two_pass_reference", "joint_masked"]

# New branches use independent fixed RNG streams so shared P1/P2 Geometry
# parameters do not depend on model mode or module-construction order.
GEOMETRY_QUERY_INIT_SEED = 2026081801
GEOMETRY_HEAD_INIT_SEED = 2026081802
GROUND_QUERY_INIT_SEED = 2026081811
GROUND_HEAD_INIT_SEED = 2026081812
MOTION_QUERY_INIT_SEED = 2026081821
MOTION_HEAD_INIT_SEED = 2026081822


@dataclasses.dataclass(frozen=True)
class PolicyAuxConfig:
    mode: PolicyAuxMode = "none"
    num_ground_queries: int = 8
    num_geometry_queries: int = 8
    num_motion_queries: int = 0
    geometry_target_dim: int = 2048
    motion_target_dim: int = 256
    motion_smooth_l1_beta: float = 1.0
    ground_mask_dim: int = 256
    ground_focal_alpha: float = 0.25
    ground_focal_gamma: float = 2.0
    lambda_sem: float | None = None
    lambda_ground: float | None = None
    lambda_geo: float | None = None
    lambda_motion: float | None = None
    diagnostic_skip_semantic_lm: bool = False
    semantic_annotation_root: str | None = None
    ground_mask_root: str | None = None
    geometry_cache_root: str | None = None
    geometry_normalization_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in (
            "none",
            "geometry",
            "semantic_geometry",
            "semantic_geometry_motion",
            "ground_geometry_semantic_lm",
        ):
            raise ValueError(f"Unsupported policy_aux_mode: {self.mode}")
        if self.diagnostic_skip_semantic_lm and self.mode != "ground_geometry_semantic_lm":
            raise ValueError("The semantic-LM ablation requires the complete P2 mode")
        if self.num_geometry_queries != 8:
            raise ValueError("Policy auxiliary Geometry query count is frozen at eight")
        expected_ground_queries = 0 if self.mode in ("semantic_geometry", "semantic_geometry_motion") else 8
        if self.num_ground_queries != expected_ground_queries:
            raise ValueError(
                f"{self.mode} requires num_ground_queries={expected_ground_queries}, found {self.num_ground_queries}"
            )
        if self.geometry_target_dim != 2048:
            raise ValueError("P1/P2 v0 Geometry target dimension is frozen at 2048")
        expected_motion_queries = 8 if self.mode == "semantic_geometry_motion" else 0
        if self.num_motion_queries != expected_motion_queries:
            raise ValueError(
                f"{self.mode} requires num_motion_queries={expected_motion_queries}, found {self.num_motion_queries}"
            )
        if self.motion_target_dim != 256 or self.motion_smooth_l1_beta != 1.0:
            raise ValueError("B Motion target dimension/beta are frozen at 256/1.0")
        for name in ("lambda_sem", "lambda_ground", "lambda_geo", "lambda_motion"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


def policy_aux_config_from_train_config(train_config) -> PolicyAuxConfig | None:
    """Translate the trainer config into the inference-safe model config.

    Teacher/cache paths intentionally do not cross this boundary: they belong to
    the training dataset, while the trained policy needs only its learned
    parameters and the frozen architectural/loss metadata.
    """

    policy_aux = train_config.policy_aux
    if policy_aux is None:
        return None
    return PolicyAuxConfig(
        mode=policy_aux.mode,
        num_ground_queries=policy_aux.num_ground_queries,
        num_geometry_queries=policy_aux.num_geometry_queries,
        num_motion_queries=getattr(policy_aux, "num_motion_queries", 0),
        ground_mask_dim=policy_aux.ground_mask_dim,
        ground_focal_alpha=policy_aux.ground_focal_alpha,
        ground_focal_gamma=policy_aux.ground_focal_gamma,
        lambda_geo=policy_aux.lambda_geo,
        lambda_ground=policy_aux.lambda_ground,
        lambda_sem=policy_aux.lambda_sem,
        lambda_motion=getattr(policy_aux, "lambda_motion", None),
        diagnostic_skip_semantic_lm=getattr(policy_aux, "diagnostic_skip_semantic_lm", False),
    )


def create_pytorch_model(train_config, *, model_config=None) -> PI0Pytorch:
    """Create the canonical PyTorch architecture for training or serving."""

    model_config = train_config.model if model_config is None else model_config
    policy_aux_config = policy_aux_config_from_train_config(train_config)
    if policy_aux_config is None:
        return PI0Pytorch(model_config)
    return PI05AuxPolicy(model_config, policy_aux_config)


def load_trained_pytorch_model(train_config, weight_path: str, *, device: str = "cpu") -> PI0Pytorch:
    """Strict-load a complete trained checkpoint through the canonical factory."""

    model = create_pytorch_model(train_config)
    missing, unexpected = safetensors.torch.load_model(model, weight_path, strict=True, device=device)
    if missing or unexpected:
        raise RuntimeError(
            f"Strict trained checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return model


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
    motion: TokenSpan | None = None
    action_suffix: TokenSpan | None = None

    @property
    def query_groups(self) -> dict[str, TokenSpan]:
        return {
            name: span
            for name, span in (
                ("geometry", self.geometry),
                ("motion", self.motion),
                ("ground", self.ground),
            )
            if span is not None
        }


@dataclasses.dataclass(frozen=True)
class JointP2TrainLayout:
    """Ephemeral P2-only layout for the joint masked training computation."""

    base_layout: PrefixLayout
    semantic: TokenSpan
    action_suffix: TokenSpan

    def __post_init__(self) -> None:
        base_end = max(
            (span.end for span in self.base_layout.query_groups.values()),
            default=self.base_layout.context.end,
        )
        if self.semantic.start != base_end:
            raise ValueError("Semantic teacher span must immediately follow the base P2 prefix")
        if self.action_suffix.start != self.semantic.end:
            raise ValueError("Action suffix must immediately follow the joint PaliGemma-side prefix")


@dataclasses.dataclass
class PolicyAuxTargets:
    geometry: torch.Tensor | None = None
    geometry_valid: torch.Tensor | None = None
    geometry_mean: torch.Tensor | None = None
    geometry_std: torch.Tensor | None = None
    motion: torch.Tensor | None = None
    motion_valid: torch.Tensor | None = None
    motion_mean: torch.Tensor | None = None
    motion_std: torch.Tensor | None = None
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


def build_joint_p2_attention(
    paligemma_pad_mask: torch.Tensor,
    suffix_pad_mask: torch.Tensor,
    suffix_ar_mask: torch.Tensor,
    layout: JointP2TrainLayout,
) -> torch.Tensor:
    """Build the P2 joint mask without exposing SemanticTeacher to Action.

    PaliGemma-side tokens are physically ``Context|Geometry|Ground|Semantic``.
    The action expert is a separate suffix. Semantic reads Context plus its own
    causal teacher prefix; Action reads only the base P2 prefix.
    """

    if paligemma_pad_mask.ndim != 2 or suffix_pad_mask.ndim != 2 or suffix_ar_mask.ndim != 2:
        raise ValueError("Joint P2 masks must be rank-2")
    if paligemma_pad_mask.shape[0] != suffix_pad_mask.shape[0]:
        raise ValueError("Joint P2 prefix/suffix batch sizes differ")
    if suffix_pad_mask.shape != suffix_ar_mask.shape:
        raise ValueError("Action suffix pad/AR masks differ")

    batch, paligemma_length = paligemma_pad_mask.shape
    suffix_length = suffix_pad_mask.shape[1]
    semantic = layout.semantic
    action = layout.action_suffix
    if semantic.end != paligemma_length:
        raise ValueError("Semantic span must end at the PaliGemma-side sequence boundary")
    if action.start != paligemma_length or action.end != paligemma_length + suffix_length:
        raise ValueError("Action suffix span is inconsistent with the joint sequence")

    base_pad = paligemma_pad_mask[:, : semantic.start].to(torch.bool)
    semantic_pad = paligemma_pad_mask[:, semantic.start : semantic.end].to(torch.bool)
    base_attention = build_explicit_aux_prefix_attention(base_pad, layout.base_layout)
    suffix_attention = make_att_2d_masks(suffix_pad_mask, suffix_ar_mask)
    full = torch.zeros(
        (batch, action.end, action.end),
        dtype=torch.bool,
        device=paligemma_pad_mask.device,
    )
    full[:, : semantic.start, : semantic.start] = base_attention

    context = layout.base_layout.context
    full[:, semantic.start : semantic.end, context.start : context.end] = (
        semantic_pad[:, :, None] & base_pad[:, None, context.start : context.end]
    )
    causal = torch.tril(
        torch.ones((semantic.length, semantic.length), dtype=torch.bool, device=paligemma_pad_mask.device)
    )
    full[:, semantic.start : semantic.end, semantic.start : semantic.end] = (
        causal[None] & semantic_pad[:, :, None] & semantic_pad[:, None, :]
    )

    suffix_valid = suffix_pad_mask.to(torch.bool)
    full[:, action.start : action.end, : semantic.start] = suffix_valid[:, :, None] & base_pad[:, None, :]
    full[:, action.start : action.end, action.start : action.end] = suffix_attention
    return full


def build_joint_p2_position_ids(
    base_prefix_pad_mask: torch.Tensor,
    semantic_pad_mask: torch.Tensor,
    suffix_pad_mask: torch.Tensor,
    layout: JointP2TrainLayout,
) -> torch.Tensor:
    """Preserve old-main and old-semantic RoPE positions in one sequence."""

    if any(mask.ndim != 2 for mask in (base_prefix_pad_mask, semantic_pad_mask, suffix_pad_mask)):
        raise ValueError("Joint P2 position masks must be rank-2")
    batch = base_prefix_pad_mask.shape[0]
    if semantic_pad_mask.shape[0] != batch or suffix_pad_mask.shape[0] != batch:
        raise ValueError("Joint P2 position-mask batch sizes differ")
    if layout.semantic.start != base_prefix_pad_mask.shape[1]:
        raise ValueError("Joint semantic span does not follow the base prefix")
    if layout.semantic.length != semantic_pad_mask.shape[1]:
        raise ValueError("Joint semantic span/mask lengths differ")
    if layout.action_suffix.length != suffix_pad_mask.shape[1]:
        raise ValueError("Joint action span/mask lengths differ")

    base_valid = base_prefix_pad_mask.to(torch.bool)
    semantic_valid = semantic_pad_mask.to(torch.bool)
    suffix_valid = suffix_pad_mask.to(torch.bool)
    main_valid = torch.cat((base_valid, suffix_valid), dim=1)
    main_positions = torch.cumsum(main_valid, dim=1) - 1

    context = layout.base_layout.context
    if context.start != 0:
        raise ValueError("Joint P2 Context must begin at zero")
    semantic_reference_valid = torch.cat((base_valid[:, : context.end], semantic_valid), dim=1)
    semantic_reference_positions = (torch.cumsum(semantic_reference_valid, dim=1) - 1).clamp_min(0)

    positions = torch.empty((batch, layout.action_suffix.end), dtype=torch.long, device=base_valid.device)
    positions[:, : layout.semantic.start] = main_positions[:, : layout.semantic.start]
    positions[:, layout.semantic.start : layout.semantic.end] = semantic_reference_positions[:, context.end :]
    positions[:, layout.action_suffix.start : layout.action_suffix.end] = main_positions[:, layout.semantic.start :]
    return positions


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

        if aux_config.mode in ("geometry", "semantic_geometry", "semantic_geometry_motion"):
            self.geometry_queries = self._new_queries(aux_config.num_geometry_queries, seed=GEOMETRY_QUERY_INIT_SEED)
            self.geometry_head = self._new_seeded_module(
                GEOMETRY_HEAD_INIT_SEED,
                lambda: MeanQueryProjectionHead(self.hidden_dim, aux_config.geometry_target_dim),
            )
            if aux_config.mode == "semantic_geometry_motion":
                self.motion_queries = self._new_queries(aux_config.num_motion_queries, seed=MOTION_QUERY_INIT_SEED)
                self.motion_head = self._new_seeded_module(
                    MOTION_HEAD_INIT_SEED,
                    lambda: MeanQueryProjectionHead(self.hidden_dim, aux_config.motion_target_dim),
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
        if self.aux_config.mode == "semantic_geometry_motion":
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
        spans: dict[str, TokenSpan | None] = {"geometry": None, "motion": None, "ground": None}

        for name in ("geometry", "motion", "ground"):
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
            motion=spans["motion"],
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

    @staticmethod
    def _validate_semantic_inputs(
        context_embeddings: torch.Tensor,
        context_pad_mask: torch.Tensor,
        language_span: TokenSpan,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return loss_mask[:, 1:].to(torch.bool), anchor_indices

    def _semantic_lm_objective(
        self,
        context_outputs: torch.Tensor,
        semantic_outputs: torch.Tensor,
        anchor_indices: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_indices = torch.arange(labels.shape[0], device=labels.device)
        prediction_states = torch.cat(
            (context_outputs[batch_indices, anchor_indices][:, None], semantic_outputs),
            dim=1,
        )
        if prediction_states.shape[:2] != labels.shape:
            raise ValueError("Semantic prediction-state/label shapes differ")
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

    def _native_semantic_lm_decode(
        self,
        context_embeddings: torch.Tensor,
        context_pad_mask: torch.Tensor,
        language_span: TokenSpan,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        attention_implementation: Literal["eager", "sdpa"] = "sdpa",
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Native autoregressive semantic text objective from image+instruction context."""

        batch, _ = labels.shape
        teacher_input_mask, anchor_indices = self._validate_semantic_inputs(
            context_embeddings,
            context_pad_mask,
            language_span,
            input_ids,
            labels,
            loss_mask,
        )

        token_embeddings = self._apply_checkpoint(self.paligemma_with_expert.embed_language_tokens, input_ids)
        token_embeddings = token_embeddings * self.hidden_dim**0.5
        decoder_embeddings = torch.cat((context_embeddings, token_embeddings), dim=1)
        attention = build_native_semantic_lm_attention(context_pad_mask, teacher_input_mask)
        valid = torch.cat((context_pad_mask.to(torch.bool), teacher_input_mask), dim=1)
        position_ids = (torch.cumsum(valid, dim=1) - 1).clamp_min(0)
        language_model = self.paligemma_with_expert.paligemma.language_model
        # The production reference defaults to SDPA: eager attention exceeds an
        # 80 GB A100 at local batch 32. Validation may explicitly request eager
        # on a tiny batch to isolate architecture from cross-kernel differences.
        language_model.config._attn_implementation = attention_implementation  # noqa: SLF001
        attention_bias = self._prepare_attention_masks_4d(attention).to(decoder_embeddings.dtype)

        def decode_semantic_language_model(
            embeddings: torch.Tensor,
            mask: torch.Tensor,
            positions: torch.Tensor,
        ) -> torch.Tensor:
            return language_model.forward(
                inputs_embeds=embeddings,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                use_cache=False,
                adarms_cond=None,
            ).last_hidden_state

        # The action/ground/geometry graph is still live at this point.  Outer
        # checkpoints prevent the additional semantic-LM graph from being
        # resident at the same time; backward recomputes this pass.  Chunking
        # only schedules that recomputation and does not change the batch-level
        # semantic objective below.
        semantic_activation_chunk_size = 8
        output_chunks = []
        for start in range(0, batch, semantic_activation_chunk_size):
            stop = min(start + semantic_activation_chunk_size, batch)
            output_chunks.append(
                self._apply_checkpoint(
                    decode_semantic_language_model,
                    decoder_embeddings[start:stop],
                    attention_bias[start:stop],
                    position_ids[start:stop],
                )
            )
        outputs = torch.cat(output_chunks, dim=0)
        result = self._semantic_lm_objective(
            outputs[:, : context_embeddings.shape[1]],
            outputs[:, context_embeddings.shape[1] :],
            anchor_indices,
            labels,
            loss_mask,
        )
        if return_hidden_states:
            result["context_hidden_states"] = outputs[:, : context_embeddings.shape[1]]
        return result

    def forward_with_aux(
        self,
        observation,
        actions: torch.Tensor,
        aux_targets: PolicyAuxTargets,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        semantic_impl: SemanticImplementation | None = None,
        reference_semantic_attention_impl: Literal["eager", "sdpa"] = "sdpa",
        return_validation_outputs: bool = False,
    ) -> dict[str, torch.Tensor | PrefixLayout | JointP2TrainLayout | None | dict[str, torch.Tensor]]:
        if not self.aux_enabled:
            raise RuntimeError("forward_with_aux requires an enabled auxiliary mode")
        if semantic_impl is None:
            # Production Semantic follows the frozen P2 implementation.  The
            # two-pass path remains available only as an explicit numerical
            # reference; Semantic+Geometry differs from P2 solely by removing
            # Ground data, queries, head, and loss.
            semantic_impl = "joint_masked"
        if semantic_impl not in ("two_pass_reference", "joint_masked"):
            raise ValueError(f"Unsupported semantic implementation: {semantic_impl}")
        if (
            self.aux_config.mode not in ("semantic_geometry", "semantic_geometry_motion", "ground_geometry_semantic_lm")
            and semantic_impl != "joint_masked"
        ):
            raise ValueError("The semantic implementation selector requires an enabled Semantic mode")

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
        base_prefix, base_prefix_pad, layout = self._append_aux_queries(
            context,
            context_pad,
            view_spans=view_spans,
            real_view_names=real_views,
            padded_view_names=padded_views,
            language_span=language_span,
        )
        suffix, suffix_pad, suffix_ar, adarms_cond = self.embed_suffix(state, x_t, time)

        semantic_enabled = (
            self.aux_config.mode
            in (
                "semantic_geometry",
                "semantic_geometry_motion",
                "ground_geometry_semantic_lm",
            )
            and not self.aux_config.diagnostic_skip_semantic_lm
        )
        teacher_input_mask = None
        semantic_anchor_indices = None
        joint_train_layout = None
        if semantic_enabled:
            required_semantic = (
                aux_targets.semantic_input_ids,
                aux_targets.semantic_labels,
                aux_targets.semantic_loss_mask,
            )
            if any(value is None for value in required_semantic):
                raise ValueError("Enabled Semantic branch requires teacher-forcing tensors")
            teacher_input_mask, semantic_anchor_indices = self._validate_semantic_inputs(
                context,
                context_pad,
                language_span,
                aux_targets.semantic_input_ids,
                aux_targets.semantic_labels,
                aux_targets.semantic_loss_mask,
            )

        if semantic_enabled and semantic_impl == "joint_masked":
            semantic_embeddings = self._apply_checkpoint(
                self.paligemma_with_expert.embed_language_tokens,
                aux_targets.semantic_input_ids,
            )
            semantic_embeddings = semantic_embeddings * self.hidden_dim**0.5
            semantic_span = TokenSpan(base_prefix.shape[1], base_prefix.shape[1] + semantic_embeddings.shape[1])
            action_span = TokenSpan(semantic_span.end, semantic_span.end + suffix.shape[1])
            joint_train_layout = JointP2TrainLayout(
                base_layout=layout,
                semantic=semantic_span,
                action_suffix=action_span,
            )
            prefix = torch.cat((base_prefix, semantic_embeddings), dim=1)
            prefix_pad = torch.cat((base_prefix_pad, teacher_input_mask), dim=1)
            attention = build_joint_p2_attention(prefix_pad, suffix_pad, suffix_ar, joint_train_layout)
            position_ids = build_joint_p2_position_ids(
                base_prefix_pad,
                teacher_input_mask,
                suffix_pad,
                joint_train_layout,
            )
        else:
            prefix = base_prefix
            prefix_pad = base_prefix_pad
            attention = build_explicit_aux_train_attention(prefix_pad, suffix_pad, suffix_ar, layout)
            all_pad = torch.cat((prefix_pad, suffix_pad), dim=1)
            position_ids = torch.cumsum(all_pad, dim=1) - 1

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix = prefix.to(torch.bfloat16)
            suffix = suffix.to(torch.bfloat16)

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
        if return_validation_outputs:
            diagnostics["action_velocity"] = action_velocity
            diagnostics["main_context_hidden_states"] = prefix_output[:, : context.shape[1]]
            diagnostics["context_pad_mask"] = context_pad

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

        if layout.motion is not None:
            motion_states = prefix_output[:, layout.motion.start : layout.motion.end]
            motion_prediction = self.motion_head(motion_states)
            required = (
                aux_targets.motion,
                aux_targets.motion_valid,
                aux_targets.motion_mean,
                aux_targets.motion_std,
            )
            if any(value is None for value in required):
                raise ValueError("Enabled Motion branch requires target, valid, mean, and std")
            losses["motion"] = masked_standardized_smooth_l1(
                motion_prediction,
                aux_targets.motion,
                aux_targets.motion_valid,
                aux_targets.motion_mean,
                aux_targets.motion_std,
                beta=self.aux_config.motion_smooth_l1_beta,
            )
            diagnostics["motion_prediction"] = motion_prediction
            diagnostics["motion_valid_count"] = aux_targets.motion_valid.to(torch.int64).sum()

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

        if semantic_enabled:
            if semantic_impl == "joint_masked":
                semantic = self._semantic_lm_objective(
                    prefix_output[:, : context.shape[1]],
                    prefix_output[:, joint_train_layout.semantic.start : joint_train_layout.semantic.end],
                    semantic_anchor_indices,
                    aux_targets.semantic_labels,
                    aux_targets.semantic_loss_mask,
                )
            else:
                semantic = self._native_semantic_lm_decode(
                    context,
                    context_pad,
                    language_span,
                    aux_targets.semantic_input_ids,
                    aux_targets.semantic_labels,
                    aux_targets.semantic_loss_mask,
                    attention_implementation=reference_semantic_attention_impl,
                    return_hidden_states=return_validation_outputs,
                )
            losses["semantic"] = semantic["loss"]
            diagnostics["semantic_token_accuracy"] = semantic["token_accuracy"]
            diagnostics["semantic_teacher_forced_exact_match"] = semantic["teacher_forced_exact_match"]
            diagnostics["semantic_supervised_token_count"] = aux_targets.semantic_loss_mask.to(torch.int64).sum()
            if return_validation_outputs:
                diagnostics["semantic_logits"] = semantic["logits"]
                diagnostics["semantic_context_hidden_states"] = (
                    prefix_output[:, : context.shape[1]]
                    if semantic_impl == "joint_masked"
                    else semantic["context_hidden_states"]
                )

        required_lambdas = {
            "geometry": self.aux_config.lambda_geo,
            "ground": self.aux_config.lambda_ground,
            "semantic": self.aux_config.lambda_sem,
            "motion": self.aux_config.lambda_motion,
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
        result = {
            "losses": losses,
            "diagnostics": diagnostics,
            "layout": dataclasses.replace(
                layout,
                action_suffix=TokenSpan(base_prefix.shape[1], base_prefix.shape[1] + suffix.shape[1]),
            ),
            "action_loss_per_element": action_loss_per_element,
        }
        if joint_train_layout is not None:
            result["joint_train_layout"] = joint_train_layout
        return result

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
