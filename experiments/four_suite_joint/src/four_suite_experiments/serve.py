"""Serve a trained LIBERO-40 auxiliary policy without loading teacher targets."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server

from .action_access import install_action_access_policy
from .configs import blocked_action_groups
from .configs import build_train_config
from .paths import ArtifactPaths
from .paths import SourcePaths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("trqc", "whole_scene", "no_query_access"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--base-weight-path", type=Path)
    parser.add_argument("--libero-assets-dir", type=Path)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()

    if args.num_steps != 10:
        raise ValueError("Formal LIBERO evaluation freezes flow-matching integration at 10 steps")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"checkpoint has no serving weights: {checkpoint}")
    if not (checkpoint / "assets/physical-intelligence/libero/norm_stats.json").is_file():
        raise FileNotFoundError(f"checkpoint has no LIBERO norm stats: {checkpoint}")

    source_paths = SourcePaths.defaults(
        args.artifact_dir, target_scope="whole_scene" if args.variant == "whole_scene" else "task_relevant"
    )
    artifacts = ArtifactPaths(source_paths.artifact_dir)
    config = build_train_config(
        variant=args.variant,
        artifacts=artifacts,
        exp_name="formal_evaluation",
        num_train_steps=30_000,
        warmup_steps=10_000,
        checkpoint_base_dir=checkpoint.parents[2],
        lerobot_root=source_paths.lerobot_root,
        base_weight_path=args.base_weight_path,
        libero_assets_dir=args.libero_assets_dir,
        wandb_enabled=False,
    )
    install_action_access_policy(blocked_action_groups(args.variant))
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device="cuda",
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    logging.info("Serving %s checkpoint %s on port %d", args.variant, checkpoint, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), force=True)
    main()
