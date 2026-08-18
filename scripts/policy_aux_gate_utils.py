"""Shared real-LIBERO sample loading for P1/P2 development gates."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import image_tools
from openpi.training.policy_aux_dataset import Libero10AnnotationIndex
from openpi.training.policy_aux_dataset import PolicySemanticTokenizer


def _decode_image(value: dict) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))


def load_real_libero_item(
    *,
    snapshot: Path,
    mapping_path: Path,
    annotation_manifest: Path,
    lerobot_episode_index: int = 0,
    frame_index: int = 0,
) -> tuple[_model.Observation, torch.Tensor, dict, pd.Series]:
    mapping = json.loads(mapping_path.read_text())
    episode = next(row for row in mapping["episodes"] if int(row["lerobot_episode_index"]) == lerobot_episode_index)
    parquet_path = snapshot / episode["parquet_relative_path"]
    frame = pd.read_parquet(parquet_path)
    current = frame.loc[frame["frame_index"].eq(frame_index)].iloc[0]

    task_rows = [json.loads(line) for line in (snapshot / "meta/tasks.jsonl").read_text().splitlines()]
    prompt = next(row["task"] for row in task_rows if int(row["task_index"]) == int(current["task_index"]))
    image = np.array(image_tools.resize_with_pad(_decode_image(current["image"]), 224, 224), copy=True)
    wrist = np.array(image_tools.resize_with_pad(_decode_image(current["wrist_image"]), 224, 224), copy=True)
    zeros = np.zeros_like(image)
    state = np.zeros(32, dtype=np.float32)
    raw_state = np.asarray(current["state"], dtype=np.float32)
    state[: len(raw_state)] = raw_state
    token_ids, token_mask = _tokenizer.PaligemmaTokenizer(200).tokenize(prompt, state=None)
    observation = _model.Observation.from_dict(
        {
            "image": {
                "base_0_rgb": torch.from_numpy(image[None]),
                "left_wrist_0_rgb": torch.from_numpy(wrist[None]),
                "right_wrist_0_rgb": torch.from_numpy(zeros[None]),
            },
            "image_mask": {
                "base_0_rgb": torch.ones(1, dtype=torch.bool),
                "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
                "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool),
            },
            "state": torch.from_numpy(state[None]),
            "tokenized_prompt": torch.from_numpy(token_ids[None].astype(np.int64)),
            "tokenized_prompt_mask": torch.from_numpy(token_mask[None]),
        }
    )

    action_rows = frame.iloc[frame_index : frame_index + 10]
    actions = np.stack(action_rows["actions"].to_numpy()).astype(np.float32)
    if len(actions) < 10:
        actions = np.concatenate((actions, np.repeat(actions[-1:], 10 - len(actions), axis=0)))
    padded_actions = np.zeros((1, 10, 32), dtype=np.float32)
    padded_actions[0, :, : actions.shape[-1]] = actions

    annotation_index = Libero10AnnotationIndex(annotation_manifest, mapping_path)
    annotation_row = annotation_index.row(lerobot_episode_index, frame_index)
    ground_masks, ground_valid = annotation_index.load_upright_ground_masks(annotation_row)
    semantic = PolicySemanticTokenizer().batch([str(annotation_row["semantic_subtask"])])
    auxiliary = {
        "sample_id": str(annotation_row["sample_id"]),
        "prompt": prompt,
        "semantic_text": str(annotation_row["semantic_subtask"]),
        "ground_masks": {key: torch.from_numpy(value[None]) for key, value in ground_masks.items()},
        "ground_valid_views": torch.from_numpy(ground_valid[None]),
        "semantic_input_ids": torch.from_numpy(semantic.input_ids),
        "semantic_labels": torch.from_numpy(semantic.labels),
        "semantic_loss_mask": torch.from_numpy(semantic.loss_mask),
    }
    return observation, torch.from_numpy(padded_actions), auxiliary, annotation_row


def move_observation(observation: _model.Observation, device: torch.device) -> _model.Observation:
    return _model.Observation(
        images={key: value.to(device) for key, value in observation.images.items()},
        image_masks={key: value.to(device) for key, value in observation.image_masks.items()},
        state=observation.state.to(device),
        tokenized_prompt=observation.tokenized_prompt.to(device),
        tokenized_prompt_mask=observation.tokenized_prompt_mask.to(device),
        token_ar_mask=None,
        token_loss_mask=None,
    )
