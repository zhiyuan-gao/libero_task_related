#!/usr/bin/env python3
"""Assert every P1/P2 explicit attention rectangle from a real prefix layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from openpi.models_pytorch.pi05_aux_queries import PrefixLayout
from openpi.models_pytorch.pi05_aux_queries import TokenSpan
from openpi.models_pytorch.pi05_aux_queries import build_explicit_aux_prefix_attention
from openpi.models_pytorch.pi05_aux_queries import build_explicit_aux_train_attention


def span(value: dict | None) -> TokenSpan | None:
    return None if value is None else TokenSpan(**value)


def rectangle(mask: torch.Tensor, rows: TokenSpan, columns: TokenSpan) -> torch.Tensor:
    return mask[rows.start : rows.end, columns.start : columns.end]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.prefix_layout.read_text())
    results = {}
    for mode, raw in payload["modes"].items():
        layout = PrefixLayout(
            view_spans={name: span(value) for name, value in raw["view_spans"].items()},
            real_view_names=tuple(raw["real_view_names"]),
            padded_view_names=tuple(raw["padded_view_names"]),
            language=span(raw["language"]),
            context=span(raw["context"]),
            ground=span(raw["ground"]),
            geometry=span(raw["geometry"]),
        )
        prefix_length = int(raw["prefix_physical_tokens"])
        pad = torch.zeros((1, prefix_length), dtype=torch.bool)
        for name in layout.real_view_names:
            view = layout.view_spans[name]
            pad[:, view.start : view.end] = True
        language_valid = (
            int(raw["prefix_valid_tokens"])
            - sum(layout.view_spans[name].length for name in layout.real_view_names)
            - sum(group.length for group in layout.query_groups.values())
        )
        pad[:, layout.language.start : layout.language.start + language_valid] = True
        for group in layout.query_groups.values():
            pad[:, group.start : group.end] = True

        prefix = build_explicit_aux_prefix_attention(pad, layout)[0]
        checks = {}
        context = layout.context
        for name, group in layout.query_groups.items():
            checks[f"context_cannot_read_{name}"] = not bool(rectangle(prefix, context, group).any())
            checks[f"{name}_reads_all_valid_context"] = bool(
                rectangle(prefix, group, context)[:, pad[0, context.start : context.end]].all()
            )
            checks[f"{name}_same_group_interaction"] = bool(rectangle(prefix, group, group).all())
        for reader_name, reader in layout.query_groups.items():
            for writer_name, writer in layout.query_groups.items():
                if reader_name != writer_name:
                    checks[f"{reader_name}_cannot_read_{writer_name}"] = not bool(
                        rectangle(prefix, reader, writer).any()
                    )
        invalid = ~pad[0]
        checks["prefix_padding_rows_fully_masked"] = not bool(prefix[invalid].any())
        checks["prefix_padding_columns_fully_masked"] = not bool(prefix[:, invalid].any())

        suffix_pad = torch.ones((1, 10), dtype=torch.bool)
        suffix_ar = torch.tensor([[1] + [0] * 9], dtype=torch.bool)
        full = build_explicit_aux_train_attention(pad, suffix_pad, suffix_ar, layout)[0]
        suffix_span = TokenSpan(prefix_length, prefix_length + 10)
        checks["prefix_cannot_read_action"] = not bool(rectangle(full, TokenSpan(0, prefix_length), suffix_span).any())
        checks["action_reads_all_valid_prefix"] = bool(
            rectangle(full, suffix_span, TokenSpan(0, prefix_length))[:, pad[0]].all()
        )
        for name, group in layout.query_groups.items():
            checks[f"action_reads_{name}"] = bool(rectangle(full, suffix_span, group).all())
        if not all(checks.values()):
            raise RuntimeError(f"{mode} attention gate failed: {checks}")
        results[mode] = {
            "checks": checks,
            "prefix_shape": list(prefix.shape),
            "full_train_shape": list(full.shape),
            "explicit_2d_connectivity": True,
        }

    output = {
        "status": "PASS",
        "gate": "pi05_p1_p2_explicit_2d_attention_connectivity_v1",
        "prefix_layout": str(args.prefix_layout.resolve()),
        "modes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
