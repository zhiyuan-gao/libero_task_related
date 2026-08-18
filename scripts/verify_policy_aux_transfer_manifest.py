#!/usr/bin/env python3
"""Verify transferred P1/P2 handoff paths against their tree digests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_policy_aux_transfer_manifest import inspect_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path("/workspace/vla"))
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    failures = []
    for entry in payload["entries"]:
        if not entry["required"]:
            continue
        destination = args.workspace_root / entry["relative_destination"]
        if not destination.exists():
            failures.append(f"missing: {destination}")
            continue
        _, digest, _ = inspect_path(destination)
        if digest != entry["sha256"]:
            failures.append(f"sha256 mismatch: {destination}: expected {entry['sha256']}, found {digest}")
    if failures:
        raise RuntimeError("Transfer verification failed:\n" + "\n".join(failures))
    print(f"PASS: verified {sum(entry['required'] for entry in payload['entries'])} required items")


if __name__ == "__main__":
    main()
