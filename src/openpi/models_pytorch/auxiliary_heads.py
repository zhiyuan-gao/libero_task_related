"""Lightweight auxiliary heads and losses for the P1/P2 policy variants."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.distributed as dist
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


def _binary_ranking_metrics(scores: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return AUROC and average precision for one detached binary patch set."""

    scores = scores.detach().float().reshape(-1)
    target = target.detach().to(torch.bool).reshape(-1)
    positive_count = target.sum()
    negative_count = (~target).sum()
    if positive_count == 0 or negative_count == 0:
        zero = scores.sum() * 0.0
        return zero, zero

    order = torch.argsort(scores, descending=True)
    sorted_target = target[order].float()
    true_positive = sorted_target.cumsum(0)
    false_positive = (1.0 - sorted_target).cumsum(0)
    recall = true_positive / positive_count
    false_positive_rate = false_positive / negative_count
    auroc = torch.trapz(
        torch.cat((torch.zeros(1, device=scores.device), recall)),
        torch.cat((torch.zeros(1, device=scores.device), false_positive_rate)),
    )
    precision = true_positive / torch.arange(1, len(scores) + 1, device=scores.device)
    average_precision = (precision * sorted_target).sum() / positive_count
    return auroc, average_precision


def grounding_fixed_balanced_binary_bce_loss(
    logits: torch.Tensor,
    target_coverage: torch.Tensor,
    valid_views: torch.Tensor,
    *,
    positive_weight: float,
    distributed_global_reduction: bool = True,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Dataset-global fixed-balanced BCE on ``m = 1[coverage > 0]``.

    The positive weight multiplies the complete per-patch BCE term. This is
    intentionally not ``BCEWithLogitsLoss(pos_weight=...)``. The numerator is
    divided by the sum of fixed patch weights over all valid views/patches.
    Under DDP, the forward scaling compensates for DDP's gradient averaging so
    the resulting gradient exactly matches one global-batch weighted mean.
    """

    if logits.shape != target_coverage.shape or logits.ndim != 3:
        raise ValueError("Ground logits and coverage targets must share shape [B,V,P]")
    if valid_views.shape != logits.shape[:2]:
        raise ValueError("Ground valid-view mask must have shape [B,V]")
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("Ground fixed positive weight must be finite and positive")

    coverage = target_coverage.to(device=logits.device, dtype=torch.float32)
    if not bool(torch.all((coverage >= 0.0) & (coverage <= 1.0))):
        raise ValueError("Patch foreground coverage must be in [0,1]")
    binary_target = coverage > 0
    valid = valid_views.to(device=logits.device, dtype=torch.bool)
    valid_patch = valid.unsqueeze(-1).expand_as(binary_target)

    per_patch = F.binary_cross_entropy_with_logits(logits.float(), binary_target.float(), reduction="none")
    patch_weight = torch.where(
        binary_target,
        torch.as_tensor(positive_weight, dtype=per_patch.dtype, device=per_patch.device),
        torch.ones((), dtype=per_patch.dtype, device=per_patch.device),
    )
    weighted_numerator = (patch_weight * per_patch)[valid_patch].sum()
    local_denominator = patch_weight[valid_patch].sum()

    if distributed_global_reduction and dist.is_available() and dist.is_initialized():
        global_denominator = local_denominator.detach().clone()
        dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
        if not bool(global_denominator > 0):
            raise ValueError("Ground batch has no valid patches across DDP ranks")
        loss = weighted_numerator * (dist.get_world_size() / global_denominator)
    else:
        if not bool(local_denominator > 0):
            return {"loss": logits.sum() * 0.0}
        loss = weighted_numerator / local_denominator

    selected_logits = logits.float()[valid_patch]
    selected_target = binary_target[valid_patch]
    raw_probability = selected_logits.sigmoid()
    calibrated_probability = (selected_logits - math.log(positive_weight)).sigmoid()
    prediction = calibrated_probability >= 0.5
    raw_prediction = raw_probability >= 0.5
    true_positive = (prediction & selected_target).sum().float()
    predicted_positive = prediction.sum().float()
    target_positive = selected_target.sum().float()
    union = (prediction | selected_target).sum().float()
    total = torch.as_tensor(selected_target.numel(), dtype=torch.float32, device=logits.device)
    positive_probability = calibrated_probability[selected_target].mean()
    negative_probability = calibrated_probability[~selected_target].mean()
    auroc, auprc = _binary_ranking_metrics(selected_logits, selected_target)

    return {
        "loss": loss,
        "bce_loss": loss.detach(),
        "fixed_positive_weight": torch.as_tensor(positive_weight, device=logits.device),
        "gt_positive_ratio": target_positive / total,
        "raw_predicted_positive_ratio": raw_prediction.float().mean(),
        "predicted_positive_ratio": prediction.float().mean(),
        "foreground_precision": (true_positive + eps) / (predicted_positive + eps),
        "foreground_recall": (true_positive + eps) / (target_positive + eps),
        "iou": (true_positive + eps) / (union + eps),
        "dice_score": (2.0 * true_positive + eps) / (predicted_positive + target_positive + eps),
        "positive_probability": positive_probability,
        "negative_probability": negative_probability,
        "probability_separation": positive_probability - negative_probability,
        "auroc": auroc,
        "auprc": auprc,
        "valid_view_count": valid.sum(),
        "binary_target": binary_target,
    }
