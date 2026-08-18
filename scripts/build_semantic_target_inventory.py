#!/usr/bin/env python3
"""Inventory exact LIBERO-10 concrete semantic targets with official tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from openpi.training.policy_aux_dataset import PolicySemanticTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-target-len", type=int, default=32)
    args = parser.parse_args()
    annotations = args.annotations.resolve(strict=True)
    frame = pd.read_parquet(annotations)
    required = {
        "suite",
        "official_task_index",
        "official_episode_index",
        "subtask_index",
        "subtask_text",
        "subtask_start_frame",
        "subtask_end_frame",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Annotation columns missing: {sorted(required - set(frame.columns))}")
    if set(frame["suite"].unique()) != {"libero_10"}:
        raise ValueError("Semantic inventory source must contain only LIBERO-10")

    tokenizer = PolicySemanticTokenizer(args.max_target_len)
    targets = []
    for text, group in frame.groupby("subtask_text", sort=True):
        token_ids = tokenizer.encode_target(str(text))
        associations = (
            group[
                [
                    "official_task_index",
                    "official_episode_index",
                    "subtask_index",
                    "subtask_start_frame",
                    "subtask_end_frame",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "official_task_index",
                    "official_episode_index",
                    "subtask_index",
                    "subtask_start_frame",
                ]
            )
        )
        targets.append(
            {
                "canonical_original_string": str(text),
                "normalization": "none",
                "token_ids_with_eos_no_bos": token_ids,
                "token_pieces_with_eos_no_bos": tokenizer.pieces(token_ids),
                "token_length": len(token_ids),
                "frame_count": len(group),
                "task_indices": sorted(int(value) for value in group["official_task_index"].unique()),
                "subtask_indices": sorted(int(value) for value in group["subtask_index"].unique()),
                "segment_associations": associations.to_dict(orient="records"),
            }
        )

    payload = {
        "status": "PASS",
        "schema": "openpi.libero10_semantic_target_inventory.v2",
        "annotations": str(annotations),
        "annotations_sha256": sha256_file(annotations),
        "frame_count": len(frame),
        "unique_target_count": len(targets),
        "tokenizer": "official OpenPI PaligemmaTokenizer sentencepiece model",
        "teacher_forcing": (
            "the final valid instruction state on the native VLM path predicts the first target "
            "token; prior GT target tokens predict the next token only inside the separate "
            "semantic LM pass; EOS supervised; no BOS; padded positions masked; no Semantic Query"
        ),
        "max_target_len": args.max_target_len,
        "maximum_observed_target_len": max(target["token_length"] for target in targets),
        "targets": targets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in ("targets",)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
