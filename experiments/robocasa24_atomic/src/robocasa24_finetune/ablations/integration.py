"""Install ablation-only model dispatch without editing the OpenPI checkout."""

from __future__ import annotations

from openpi.models_pytorch import pi05_aux_queries as aux_model

from ..integration import install_robocasa_overlays
from .model import RoboCasaAblationPolicy
from .model import build_ablation_aux_train_attention
from .model import build_ablation_joint_attention
from .model import build_ablation_joint_position_ids
from .specs import get_ablation_spec

_INSTALLED = False


def install_ablation_overlays() -> None:
    """Install generic RoboCasa data dispatch plus ablation model dispatch."""

    global _INSTALLED  # noqa: PLW0603
    if _INSTALLED:
        return
    install_robocasa_overlays()
    main_factory = aux_model.create_pytorch_model

    def create_pytorch_model(train_config, *, model_config=None):
        model_config = train_config.model if model_config is None else model_config
        policy_aux = train_config.policy_aux
        ablation = getattr(policy_aux, "ablation_variant", None)
        if ablation is None:
            return main_factory(train_config, model_config=model_config)
        expected_name = f"_ablation_{ablation}"
        if expected_name not in str(train_config.name):
            raise ValueError(
                "ablation metadata and training config name do not match"
            )
        spec = get_ablation_spec(str(ablation))
        policy_aux_config = aux_model.policy_aux_config_from_train_config(train_config)
        if policy_aux_config is None:
            raise ValueError("ablation training requires an auxiliary config")
        return RoboCasaAblationPolicy(model_config, policy_aux_config, spec)

    aux_model.create_pytorch_model = create_pytorch_model
    aux_model.build_explicit_aux_train_attention = build_ablation_aux_train_attention
    aux_model.build_joint_p2_attention = build_ablation_joint_attention
    aux_model.build_joint_p2_position_ids = build_ablation_joint_position_ids
    _INSTALLED = True
