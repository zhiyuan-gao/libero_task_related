"""Serve a RoboCasa checkpoint with the matching training-time transforms."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import socket
from typing import Any

import torch

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server

from .configs import build_train_config
from .eval_protocol import FLOW_STEPS
from .integration import install_robocasa_overlays


def _metadata(checkpoint: Path) -> dict[str, Any]:
    metadata_path = checkpoint / "metadata.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    payload = torch.load(metadata_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"invalid checkpoint metadata: {metadata_path}")
    return payload


def _path_arg(override: str | None, original: Any, name: str) -> str:
    value = override if override is not None else original
    if not value:
        raise ValueError(f"checkpoint metadata does not provide {name}; pass --{name.replace('_', '-')}")
    return str(value)


def build_policy(args: argparse.Namespace):
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    payload = _metadata(checkpoint)
    saved = payload["config"]
    saved_data = saved.get("data", {})
    saved_aux = saved.get("policy_aux")
    saved_policy_metadata = saved.get("policy_metadata", {})
    variant = str(saved_policy_metadata.get("variant", ""))
    tasks = tuple(saved_policy_metadata.get("task_names", ()))
    if variant not in ("baseline", "task_relevant", "whole_scene"):
        raise ValueError(f"checkpoint has an unknown RoboCasa variant: {variant!r}")
    if not tasks:
        raise ValueError("checkpoint does not record its RoboCasa task population")
    artifact_dir = None
    if variant != "baseline":
        artifact_dir = _path_arg(
            args.artifact_dir,
            saved_aux.get("artifact_dir") if isinstance(saved_aux, dict) else None,
            "artifact_dir",
        )

    policy_assets = saved_data.get("assets", {})
    config = build_train_config(
        variant=variant,
        exp_name=f"eval_{checkpoint.parent.name}_{checkpoint.name}",
        data_root=_path_arg(args.data_root, saved_data.get("data_root"), "data_root"),
        manifest_root=_path_arg(
            args.manifest_root, saved_data.get("manifest_root"), "manifest_root"
        ),
        policy_assets_root=_path_arg(
            args.policy_assets_root,
            policy_assets.get("assets_dir") if isinstance(policy_assets, dict) else None,
            "policy_assets_root",
        ),
        artifact_dir=artifact_dir,
        base_weight_dir=_path_arg(
            args.base_weight_dir, saved.get("pytorch_weight_path"), "base_weight_dir"
        ),
        checkpoint_base_dir=str(checkpoint.parent.parent),
        seed=int(saved.get("seed", 42)),
        wandb_enabled=False,
        tasks=tasks,
    )
    install_robocasa_overlays()
    return policy_config.create_trained_policy(
        config,
        checkpoint,
        sample_kwargs={"num_steps": args.flow_steps},
        pytorch_device="cuda",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--flow-steps", type=int, default=FLOW_STEPS)
    parser.add_argument("--data-root")
    parser.add_argument("--manifest-root")
    parser.add_argument("--policy-assets-root")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--base-weight-dir")
    args = parser.parse_args()
    if args.flow_steps != FLOW_STEPS:
        raise ValueError(f"formal RoboCasa inference requires {FLOW_STEPS} flow steps")
    policy = build_policy(args)
    hostname = socket.gethostname()
    logging.info("serving RoboCasa policy on %s:%d", hostname, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

