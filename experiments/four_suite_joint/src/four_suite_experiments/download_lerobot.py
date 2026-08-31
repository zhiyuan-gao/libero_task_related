"""Download and validate the frozen official four-suite LeRobot snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from .constants import LIBERO_REPO_ID
from .constants import LIBERO_REVISION
from .validate import validate_lerobot_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Hugging Face Hub cache directory on persistent HPC storage.",
    )
    args = parser.parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=LIBERO_REPO_ID,
            repo_type="dataset",
            revision=LIBERO_REVISION,
            cache_dir=str(cache_dir),
        )
    ).resolve()
    report = validate_lerobot_snapshot(snapshot, require_complete=True)
    report.update(
        {
            "status": "PASS",
            "hf_repo_id": LIBERO_REPO_ID,
            "hf_revision": LIBERO_REVISION,
            "downloaded_snapshot": str(snapshot),
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
