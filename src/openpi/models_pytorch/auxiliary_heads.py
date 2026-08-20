"""Lightweight auxiliary heads and losses for the P1/P2 policy variants."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


class MeanQueryProjectionHead(nn.Module):
    """Per-query LayerNorm, mean pooling, then a linear projection."""

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, query_states: torch.Tensor) -> torch.Tensor:
        if query_states.ndim != 3:
            raise ValueError(f"Expected [B,Q,D] query states, got {tuple(query_states.shape)}")
        return self.output_projection(self.query_norm(query_states.float()).mean(dim=1))


class QueryConditionedPatchMaskHead(nn.Module):
    """Small query/patch similarity decoder shared by the real policy image views."""

    def __init__(self, hidden_dim: int, mask_dim: int = 256) -> None:
        super().__init__()
        if mask_dim <= 0:
            raise ValueError("mask_dim must be positive")
        self.hidden_dim = hidden_dim
        self.mask_dim = mask_dim
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.patch_norm = nn.LayerNorm(hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, mask_dim)
        self.patch_projection = nn.Linear(hidden_dim, mask_dim)
        self.logit_bias = nn.Parameter(torch.zeros(()))

    def forward(self, query_states: torch.Tensor, patch_states: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[B,V,P]``.

        ``query_states`` has shape ``[B,Q,D]`` and ``patch_states`` has shape
        ``[B,V,P,D]``. The eight Grounding queries are latent capacity, not
        predefined object slots, so they are normalized and mean-pooled.
        """

        if query_states.ndim != 3 or patch_states.ndim != 4:
            raise ValueError("Expected query [B,Q,D] and patch [B,V,P,D] tensors")
        if query_states.shape[0] != patch_states.shape[0]:
            raise ValueError("Query and patch batch sizes differ")
        if query_states.shape[-1] != self.hidden_dim or patch_states.shape[-1] != self.hidden_dim:
            raise ValueError("Unexpected hidden dimension")

        query = self.query_projection(self.query_norm(query_states.float()).mean(dim=1))
        patches = self.patch_projection(self.patch_norm(patch_states.float()))
        return torch.einsum("bd,bvpd->bvp", query, patches) / math.sqrt(self.mask_dim) + self.logit_bias


def masked_standardized_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Train-standardized MSE averaged over valid samples only."""

    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shapes differ: {prediction.shape} vs {target.shape}")
    if prediction.ndim != 2:
        raise ValueError("Geometry tensors must have shape [B,D]")
    valid = valid.to(device=prediction.device, dtype=torch.bool)
    if valid.shape != prediction.shape[:1]:
        raise ValueError("Geometry valid mask must have shape [B]")
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    safe_std = std.to(device=prediction.device, dtype=torch.float32)
    if not bool(torch.all(safe_std > 0)):
        raise ValueError("Geometry standard deviations must be positive")
    center = mean.to(device=prediction.device, dtype=torch.float32)
    standardized_prediction = (prediction.float() - center) / safe_std
    standardized_target = (target.float() - center) / safe_std
    return F.mse_loss(standardized_prediction[valid], standardized_target[valid])


def masked_standardized_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Train-standardized Smooth-L1 averaged over valid samples and dimensions."""

    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shapes differ: {prediction.shape} vs {target.shape}")
    if prediction.ndim != 2:
        raise ValueError("Motion tensors must have shape [B,D]")
    valid = valid.to(device=prediction.device, dtype=torch.bool)
    if valid.shape != prediction.shape[:1]:
        raise ValueError("Motion valid mask must have shape [B]")
    if beta <= 0:
        raise ValueError("Motion Smooth-L1 beta must be positive")
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    safe_std = std.to(device=prediction.device, dtype=torch.float32)
    if not bool(torch.all(safe_std > 0)):
        raise ValueError("Motion standard deviations must be positive")
    center = mean.to(device=prediction.device, dtype=torch.float32)
    standardized_prediction = (prediction.float() - center) / safe_std
    standardized_target = (target.float() - center) / safe_std
    return F.smooth_l1_loss(standardized_prediction[valid], standardized_target[valid], beta=beta)


def grounding_focal_dice_loss(
    logits: torch.Tensor,
    target_coverage: torch.Tensor,
    valid_views: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Valid-view sigmoid focal plus soft-Dice loss for patch coverage targets."""

    if logits.shape != target_coverage.shape or logits.ndim != 3:
        raise ValueError("Ground logits and coverage targets must share shape [B,V,P]")
    if valid_views.shape != logits.shape[:2]:
        raise ValueError("Ground valid-view mask must have shape [B,V]")
    if not 0.0 <= alpha <= 1.0 or gamma < 0.0:
        raise ValueError("Invalid focal parameters")

    target = target_coverage.to(device=logits.device, dtype=torch.float32)
    if not bool(torch.all((target >= 0.0) & (target <= 1.0))):
        raise ValueError("Patch foreground coverage must be in [0,1]")
    valid = valid_views.to(device=logits.device, dtype=torch.bool)
    probabilities = logits.float().sigmoid()

    ce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    p_t = probabilities * target + (1.0 - probabilities) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal_per_view = (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean(dim=-1)

    intersection = (probabilities * target).sum(dim=-1)
    dice_score = (2.0 * intersection + eps) / (probabilities.sum(dim=-1) + target.sum(dim=-1) + eps)
    dice_loss_per_view = 1.0 - dice_score

    if bool(valid.any()):
        focal = focal_per_view[valid].mean()
        dice_loss = dice_loss_per_view[valid].mean()
        dice = dice_score[valid].mean()
        binary_prediction = probabilities >= 0.5
        binary_target = target > 0.0
        true_positive = (binary_prediction & binary_target).sum(dim=-1).float()
        predicted_positive = binary_prediction.sum(dim=-1).float()
        target_positive = binary_target.sum(dim=-1).float()
        precision = ((true_positive + eps) / (predicted_positive + eps))[valid].mean()
        recall = ((true_positive + eps) / (target_positive + eps))[valid].mean()
        union = (binary_prediction | binary_target).sum(dim=-1).float()
        iou = ((true_positive + eps) / (union + eps))[valid].mean()
    else:
        zero = logits.sum() * 0.0
        focal = zero
        dice_loss = zero
        dice = zero
        precision = zero
        recall = zero
        iou = zero

    def mean_by_view(values: torch.Tensor) -> torch.Tensor:
        counts = valid.sum(dim=0)
        means = (values * valid).sum(dim=0) / counts.clamp_min(1)
        return torch.where(counts > 0, means, torch.zeros_like(means))

    return {
        "loss": focal + dice_loss,
        "focal_loss": focal,
        "dice_loss": dice_loss,
        "dice_score": dice,
        "iou": iou,
        "foreground_precision": precision,
        "foreground_recall": recall,
        "valid_view_count": valid.sum(),
        "valid_count_by_view": valid.sum(dim=0),
        "focal_loss_by_view": mean_by_view(focal_per_view),
        "dice_loss_by_view": mean_by_view(dice_loss_per_view),
        "dice_score_by_view": mean_by_view(dice_score),
        "precision_by_view": mean_by_view((true_positive + eps) / (predicted_positive + eps))
        if bool(valid.any())
        else torch.zeros(logits.shape[1], device=logits.device),
        "recall_by_view": mean_by_view((true_positive + eps) / (target_positive + eps))
        if bool(valid.any())
        else torch.zeros(logits.shape[1], device=logits.device),
        "iou_by_view": mean_by_view((true_positive + eps) / (union + eps))
        if bool(valid.any())
        else torch.zeros(logits.shape[1], device=logits.device),
    }
