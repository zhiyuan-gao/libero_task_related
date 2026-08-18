"""Policy-image preprocessing with synchronized Grounding-mask geometry.

This module intentionally mirrors ``preprocess_observation_pytorch``.  It is
used only by the P2 auxiliary path; the disabled/P0 and P1 paths continue to
call the official function directly.  Color jitter is applied only to RGB,
while resize/crop/rotation parameters are shared exactly with the matching
continuous mask-coverage tensor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses

import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS
from openpi.models_pytorch.preprocessing_pytorch import IMAGE_RESOLUTION
from openpi.shared import image_tools


@dataclasses.dataclass(frozen=True)
class ViewGeometryRecord:
    input_height: int
    input_width: int
    output_height: int
    output_width: int
    crop_top: int | None
    crop_left: int | None
    crop_height: int | None
    crop_width: int | None
    rotation_degrees: float | None


@dataclasses.dataclass(frozen=True)
class SynchronizedPreprocessingResult:
    observation: object
    ground_masks: dict[str, torch.Tensor]
    geometry_records: dict[str, ViewGeometryRecord]


def raw_opengl_mask_to_policy_canvas(
    raw_mask: torch.Tensor,
    *,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> torch.Tensor:
    """Rotate a canonical raw LIBERO mask 180 degrees and resize with zero pad.

    Canonical annotation masks remain untouched on disk.  This function creates
    the policy-space float coverage target corresponding to the already-upright
    official LeRobot RGB frame.
    """

    if raw_mask.ndim not in (2, 3):
        raise ValueError("raw_mask must have shape [H,W] or [B,H,W]")
    batched = raw_mask.ndim == 3
    coverage = raw_mask.float()
    if not batched:
        coverage = coverage[None]
    coverage = torch.flip(coverage, dims=(-2, -1))
    coverage = _resize_coverage_with_pad(coverage, *image_resolution)
    return coverage if batched else coverage[0]


def _resize_coverage_with_pad(
    coverage: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if coverage.ndim != 3:
        raise ValueError("coverage must have shape [B,H,W]")
    current_height, current_width = coverage.shape[-2:]
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = F.interpolate(
        coverage[:, None].float(),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    return F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), value=0.0).clamp_(0.0, 1.0)


def _rotation_grid(
    batch: int,
    height: int,
    width: int,
    angle_degrees: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    angle_radians = angle_degrees * torch.pi / 180.0
    cosine = torch.cos(angle_radians)
    sine = torch.sin(angle_radians)
    grid_x = torch.linspace(-1, 1, width, device=device)
    grid_y = torch.linspace(-1, 1, height, device=device)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(batch, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(batch, -1, -1)
    rotated_x = grid_x * cosine - grid_y * sine
    rotated_y = grid_x * sine + grid_y * cosine
    return torch.stack((rotated_x, rotated_y), dim=-1)


def preprocess_observation_and_ground_masks_pytorch(
    observation,
    ground_masks: Mapping[str, torch.Tensor],
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    generator: torch.Generator | None = None,
) -> SynchronizedPreprocessingResult:
    """Apply official policy preprocessing and the identical RGB/mask geometry."""

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")
    unknown_masks = set(ground_masks) - set(image_keys)
    if unknown_masks:
        raise ValueError(f"Grounding masks refer to unknown image views: {sorted(unknown_masks)}")

    batch_shape = observation.state.shape[:-1]
    out_images: dict[str, torch.Tensor] = {}
    out_ground_masks: dict[str, torch.Tensor] = {}
    records: dict[str, ViewGeometryRecord] = {}

    for key in image_keys:
        image = observation.images[key]
        channels_first = image.shape[1] == 3
        if channels_first:
            image = image.permute(0, 2, 3, 1)
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape for {key}: {tuple(image.shape)}")
        batch, input_height, input_width, _ = image.shape

        coverage = ground_masks.get(key)
        if coverage is not None:
            coverage = coverage.to(device=image.device, dtype=torch.float32)
            if coverage.ndim == 4 and coverage.shape[1] == 1:
                coverage = coverage[:, 0]
            if coverage.ndim != 3 or coverage.shape[0] != batch:
                raise ValueError(f"Ground mask for {key} must have shape [B,H,W]")

        if image.shape[1:3] != image_resolution:
            image = image_tools.resize_with_pad_torch(image, *image_resolution)
        if coverage is not None and coverage.shape[-2:] != image_resolution:
            coverage = _resize_coverage_with_pad(coverage, *image_resolution)

        crop_top = crop_left = crop_height = crop_width = None
        rotation_degrees = None
        if train:
            image = image / 2.0 + 0.5
            if "wrist" not in key:
                height, width = image.shape[1:3]
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)
                max_h = height - crop_height
                max_w = width - crop_width
                crop_top = int(torch.randint(0, max_h + 1, (1,), device=image.device, generator=generator).item())
                crop_left = int(torch.randint(0, max_w + 1, (1,), device=image.device, generator=generator).item())
                image = image[:, crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]
                image = F.interpolate(
                    image.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
                if coverage is not None:
                    coverage = coverage[:, crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]
                    coverage = F.interpolate(
                        coverage[:, None], size=(height, width), mode="bilinear", align_corners=False
                    )[:, 0]

                angle = torch.rand(1, device=image.device, generator=generator) * 10 - 5
                rotation_degrees = float(angle.item())
                if bool(torch.abs(angle) > 0.1):
                    grid = _rotation_grid(batch, height, width, angle, device=image.device)
                    image = F.grid_sample(
                        image.permute(0, 3, 1, 2),
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    if coverage is not None:
                        coverage = F.grid_sample(
                            coverage[:, None],
                            grid,
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=False,
                        )[:, 0]

            brightness = 0.7 + torch.rand(1, device=image.device, generator=generator) * 0.6
            image = image * brightness
            contrast = 0.6 + torch.rand(1, device=image.device, generator=generator) * 0.8
            mean = image.mean(dim=(1, 2, 3), keepdim=True)
            image = (image - mean) * contrast + mean
            saturation = 0.5 + torch.rand(1, device=image.device, generator=generator)
            gray = image.mean(dim=-1, keepdim=True)
            image = (gray + (image - gray) * saturation).clamp(0, 1)
            image = image * 2.0 - 1.0

        if coverage is not None:
            out_ground_masks[key] = coverage.clamp(0.0, 1.0)
        if channels_first:
            image = image.permute(0, 3, 1, 2)
        out_images[key] = image
        records[key] = ViewGeometryRecord(
            input_height=input_height,
            input_width=input_width,
            output_height=image_resolution[0],
            output_width=image_resolution[1],
            crop_top=crop_top,
            crop_left=crop_left,
            crop_height=crop_height,
            crop_width=crop_width,
            rotation_degrees=rotation_degrees,
        )

    out_masks = {
        key: observation.image_masks.get(
            key, torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        )
        for key in out_images
    }

    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for name, value in kwargs.items():
                setattr(self, name, value)

    processed = SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
    return SynchronizedPreprocessingResult(processed, out_ground_masks, records)


def patch_foreground_coverage(
    transformed_mask: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
) -> torch.Tensor:
    """Average continuous foreground coverage into the runtime image-token grid."""

    if transformed_mask.ndim != 3:
        raise ValueError("transformed_mask must have shape [B,H,W]")
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("Patch-grid dimensions must be positive")
    pooled = F.adaptive_avg_pool2d(transformed_mask[:, None].float(), (grid_height, grid_width))
    return pooled[:, 0].flatten(1).clamp_(0.0, 1.0)
