"""RoboCasa-to-openpi policy transforms.

The pi0.5 base model retains its native 32-D state/action interface. These
transforms expose the real 16-D state and 12-D action; the standard OpenPI
model transform pads both to 32 dimensions after normalization.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as model_api

from .constants import ACTION_DIM
from .constants import POLICY_VIEW_NAMES
from .constants import STATE_DIM


def _parse_image(value: object) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"RoboCasa image must be uint8[H,W,3], found {image.shape}/{image.dtype}"
        )
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class RoboCasaInputs(transforms.DataTransformFn):
    """Map the prepared HDF5 policy row to the canonical three-view input."""

    model_type: model_api.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"RoboCasa state must be finite float32[{STATE_DIM}]")
        images = {
            "base_0_rgb": _parse_image(data["observation/image_left"]),
            "left_wrist_0_rgb": _parse_image(data["observation/wrist_image"]),
            "right_wrist_0_rgb": _parse_image(data["observation/image_right"]),
        }
        if tuple(images) != POLICY_VIEW_NAMES:
            raise AssertionError("RoboCasa policy view order changed")
        result = {
            "state": state,
            "image": images,
            "image_mask": dict.fromkeys(POLICY_VIEW_NAMES, np.True_),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if (
                actions.ndim != 2
                or actions.shape[1] != ACTION_DIM
                or not np.isfinite(actions).all()
            ):
                raise ValueError(
                    f"RoboCasa actions must be finite float32[T,{ACTION_DIM}]"
                )
            result["actions"] = actions
        if "prompt" in data:
            prompt = str(data["prompt"])
            if not prompt:
                raise ValueError("RoboCasa prompt must be non-empty")
            result["prompt"] = prompt
        return result


@dataclasses.dataclass(frozen=True)
class RoboCasaOutputs(transforms.DataTransformFn):
    """Return only the real 12 RoboCasa controls after denormalization."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < ACTION_DIM:
            raise ValueError(
                "Model action output is narrower than the RoboCasa action space"
            )
        return {"actions": actions[..., :ACTION_DIM]}
