#!/usr/bin/env python3
"""Introspect P1/P2 prefix spans from official embed paths and a real LIBERO batch."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from policy_aux_gate_utils import load_real_libero_item
from policy_aux_gate_utils import move_observation
import torch

from openpi.models import pi0_config
from openpi.models_pytorch.pi05_aux_queries import PI05AuxPolicy
from openpi.models_pytorch.pi05_aux_queries import PolicyAuxConfig
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def span_dict(span) -> dict[str, int] | None:
    return None if span is None else dataclasses.asdict(span)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--annotation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    observation, actions, auxiliary, _ = load_real_libero_item(
        snapshot=args.snapshot,
        mapping_path=args.mapping,
        annotation_manifest=args.annotation_manifest,
    )
    observation = move_observation(observation, device)
    actions = actions.to(device)
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        pytorch_compile_mode=None,
    )
    modes = {}
    for mode in ("geometry", "ground_geometry_semantic_lm"):
        model = PI05AuxPolicy(config, PolicyAuxConfig(mode=mode))
        load_result = model.load_official_base_checkpoint(str(args.checkpoint), device="cpu")
        model.to(device).eval()
        processed = _preprocessing.preprocess_observation_pytorch(observation, train=False)
        with torch.no_grad():
            context, context_pad, view_spans, real_views, padded_views, language_span = (
                model._embed_context_with_layout(  # noqa: SLF001
                    processed.images,
                    processed.image_masks,
                    processed.tokenized_prompt,
                    processed.tokenized_prompt_mask,
                )
            )
            prefix, prefix_pad, layout = model._append_aux_queries(  # noqa: SLF001
                context,
                context_pad,
                view_spans=view_spans,
                real_view_names=real_views,
                padded_view_names=padded_views,
                language_span=language_span,
            )
            suffix, _, _, _ = model.embed_suffix(
                processed.state,
                torch.zeros_like(actions),
                torch.full((1,), 0.5, device=device),
            )
        modes[mode] = {
            "strict_load": load_result,
            "view_order": list(processed.images),
            "view_spans": {name: span_dict(span) for name, span in layout.view_spans.items()},
            "tokens_per_view": {name: span.length for name, span in layout.view_spans.items()},
            "real_view_names": list(layout.real_view_names),
            "padded_view_names": list(layout.padded_view_names),
            "language": span_dict(layout.language),
            "context": span_dict(layout.context),
            "ground": span_dict(layout.ground),
            "geometry": span_dict(layout.geometry),
            "action_suffix": {"start": prefix.shape[1], "end": prefix.shape[1] + suffix.shape[1]},
            "prefix_valid_tokens": int(prefix_pad.sum()),
            "prefix_physical_tokens": int(prefix.shape[1]),
        }
        del model
        torch.cuda.empty_cache()

    p2 = modes["ground_geometry_semantic_lm"]
    checks = {
        "official_view_order": p2["view_order"] == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        "real_agent_and_wrist": p2["real_view_names"] == ["base_0_rgb", "left_wrist_0_rgb"],
        "right_wrist_is_padded": p2["padded_view_names"] == ["right_wrist_0_rgb"],
        "all_image_spans_runtime_derived": all(value == 256 for value in p2["tokens_per_view"].values()),
        "p2_query_order": (
            p2["context"]["end"] == p2["geometry"]["start"]
            and p2["geometry"]["end"] == p2["ground"]["start"]
            and p2["ground"]["end"] == p2["action_suffix"]["start"]
        ),
        "p2_has_no_semantic_query": "semantic" not in p2,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Prefix-layout gate failed: {checks}")
    payload = {
        "status": "PASS",
        "gate": "pi05_p1_p2_real_libero_prefix_layout_v1",
        "sample_id": auxiliary["sample_id"],
        "prompt": auxiliary["prompt"],
        "checks": checks,
        "modes": modes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
