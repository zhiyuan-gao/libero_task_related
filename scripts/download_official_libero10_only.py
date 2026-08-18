#!/usr/bin/env python3
"""Download only the 379 canonical official LIBERO-10 policy episodes."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()
    patterns = ["meta/*"] + [f"data/chunk-000/episode_{episode_index:06d}.parquet" for episode_index in range(379)]
    path = snapshot_download(
        repo_id="physical-intelligence/libero",
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=patterns,
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
    )
    print(path)


if __name__ == "__main__":
    main()
