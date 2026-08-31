"""Runtime-only OpenPI overlays for RoboCasa training.

Nothing in this module edits the imported OpenPI checkout. The substitutions
exist only inside a process launched through this experiment package.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from openpi import transforms as openpi_transforms
from openpi.models import model as model_api
from openpi.models_pytorch import pi05_aux_queries as aux_model
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config as openpi_config
from openpi.training import data_loader as openpi_data
from openpi.training import policy_aux_dataset as openpi_aux_data

from .auxiliary import RoboCasaPolicyAuxTransformedDataset
from .constants import ACTION_HORIZON
from .constants import DATASET_REPO_ID
from .constants import TASKS
from .data import RoboCasa24HDF5Dataset
from .policy import RoboCasaInputs
from .policy import RoboCasaOutputs


@dataclasses.dataclass(frozen=True)
class RoboCasaRuntimeDataConfig(openpi_config.DataConfig):
    data_root: str = ""
    manifest_root: str = ""
    tasks: tuple[str, ...] = TASKS
    sampling_seed: int = 42


@dataclasses.dataclass(frozen=True)
class RoboCasaDataConfigFactory(openpi_config.DataConfigFactory):
    data_root: str = ""
    manifest_root: str = ""
    tasks: tuple[str, ...] = TASKS
    sampling_seed: int = 42

    def create(
        self, assets_dirs: Path, model_config: model_api.BaseModelConfig
    ) -> RoboCasaRuntimeDataConfig:
        if not self.data_root or not self.manifest_root:
            raise ValueError("RoboCasa data and source-manifest roots are required")
        base = self.create_base_config(assets_dirs, model_config)
        values = {
            field.name: getattr(base, field.name)
            for field in dataclasses.fields(openpi_config.DataConfig)
        }
        values.update(
            data_root=str(Path(self.data_root).resolve(strict=True)),
            manifest_root=str(Path(self.manifest_root).resolve(strict=True)),
            tasks=tuple(self.tasks),
            sampling_seed=int(self.sampling_seed),
            repack_transforms=openpi_transforms.Group(),
            data_transforms=openpi_transforms.Group(
                inputs=[RoboCasaInputs(model_type=model_config.model_type)],
                outputs=[RoboCasaOutputs()],
            ),
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
            prompt_from_task=False,
            # The RoboCasa OpenPI fork keeps DataConfig's default z-score
            # normalization. Our shared local OpenPI checkout selects
            # quantiles automatically for pi0.5 because it also serves the
            # LIBERO experiments, so pin RoboCasa semantics explicitly.
            use_quantile_norm=False,
        )
        return RoboCasaRuntimeDataConfig(**values)


class RoboCasaPI05AuxPolicy(aux_model.PI05AuxPolicy):
    """The reviewed auxiliary policy with a 50-token action suffix.

    The existing implementation has a LIBERO-only constructor assertion for a
    10-step horizon. Horizon does not parameterize any trainable module. We run
    that constructor with an otherwise identical proxy, then restore the real
    50-step immutable model config before any forward or checkpoint operation.
    """

    def __init__(self, config, aux_config) -> None:
        if config.pi05 is not True or config.discrete_state_input is not False:
            raise ValueError(
                "RoboCasa auxiliary training requires continuous-state pi0.5"
            )
        if config.action_horizon != ACTION_HORIZON:
            raise ValueError(f"RoboCasa action horizon must be {ACTION_HORIZON}")
        proxy = dataclasses.replace(
            config, action_horizon=10, pytorch_compile_mode=None
        )
        super().__init__(proxy, aux_config)
        self.config = config


_INSTALLED = False
_ORIGINAL_CREATE_TORCH_DATASET = openpi_data.create_torch_dataset
_ORIGINAL_POLICY_AUX_DATASET = openpi_aux_data.PolicyAuxTransformedDataset
_ORIGINAL_CREATE_MODEL = aux_model.create_pytorch_model


def install_robocasa_overlays() -> None:
    """Install process-local data/model dispatch before trainer construction."""

    global _INSTALLED  # noqa: PLW0603
    if _INSTALLED:
        return

    def create_torch_dataset(
        data_config, action_horizon, model_config, *, policy_aux_config=None
    ):
        if isinstance(data_config, RoboCasaRuntimeDataConfig):
            if data_config.repo_id != DATASET_REPO_ID:
                raise ValueError(
                    "RoboCasa runtime data config uses the wrong repo identity"
                )
            return RoboCasa24HDF5Dataset(
                data_config.data_root,
                data_config.manifest_root,
                action_horizon=action_horizon,
                tasks=tuple(data_config.tasks),
                sampling_seed=data_config.sampling_seed,
            )
        return _ORIGINAL_CREATE_TORCH_DATASET(
            data_config,
            action_horizon,
            model_config,
            policy_aux_config=policy_aux_config,
        )

    def create_pytorch_model(train_config, *, model_config=None):
        model_config = train_config.model if model_config is None else model_config
        if not str(train_config.name).startswith("pi05_robocasa"):
            return _ORIGINAL_CREATE_MODEL(train_config, model_config=model_config)
        policy_aux = aux_model.policy_aux_config_from_train_config(train_config)
        if policy_aux is None:
            return PI0Pytorch(model_config)
        return RoboCasaPI05AuxPolicy(model_config, policy_aux)

    openpi_data.create_torch_dataset = create_torch_dataset
    openpi_aux_data.PolicyAuxTransformedDataset = RoboCasaPolicyAuxTransformedDataset
    aux_model.create_pytorch_model = create_pytorch_model
    _INSTALLED = True
