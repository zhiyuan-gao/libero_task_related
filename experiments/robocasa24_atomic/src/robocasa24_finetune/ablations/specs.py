"""Single source of truth for the six matched paper ablations."""

from __future__ import annotations

import dataclasses
from typing import Literal

AblationVariant = Literal[
    "geometry_only",
    "semantic_geometry",
    "semantic_motion",
    "full",
    "supervision_only",
    "whole_scene",
]
TargetScope = Literal["task_relevant", "whole_scene"]
InternalMode = Literal["geometry", "semantic_geometry", "semantic_geometry_motion"]
QueryGroup = Literal["geometry", "motion"]


@dataclasses.dataclass(frozen=True)
class AblationSpec:
    """Scientific differences allowed inside one otherwise frozen recipe."""

    variant: AblationVariant
    display_name: str
    target_scope: TargetScope
    internal_mode: InternalMode
    semantic_enabled: bool
    geometry_enabled: bool
    motion_enabled: bool
    action_conditioning: tuple[QueryGroup, ...]
    lambda_geo: float | None
    lambda_sem: float | None
    lambda_motion: float | None

    def __post_init__(self) -> None:
        enabled_groups = tuple(
            name
            for name, enabled in (
                ("geometry", self.geometry_enabled),
                ("motion", self.motion_enabled),
            )
            if enabled
        )
        if any(name not in enabled_groups for name in self.action_conditioning):
            raise ValueError(
                f"{self.variant}: Action cannot read a disabled auxiliary query"
            )
        if len(set(self.action_conditioning)) != len(self.action_conditioning):
            raise ValueError(f"{self.variant}: repeated Action-conditioning query")
        expected_lambdas = {
            "geometry": 0.05 if self.geometry_enabled else None,
            "semantic": 0.01 if self.semantic_enabled else None,
            "motion": 0.05 if self.motion_enabled else None,
        }
        observed_lambdas = {
            "geometry": self.lambda_geo,
            "semantic": self.lambda_sem,
            "motion": self.lambda_motion,
        }
        if observed_lambdas != expected_lambdas:
            raise ValueError(
                f"{self.variant}: loss coefficients differ from the matched protocol"
            )
        if self.internal_mode == "geometry" and enabled_groups != ("geometry",):
            raise ValueError("geometry mode must contain only Geometry")
        if self.internal_mode == "semantic_geometry" and (
            not self.semantic_enabled or enabled_groups != ("geometry",)
        ):
            raise ValueError("semantic_geometry mode requires Semantic and Geometry")
        if self.internal_mode == "semantic_geometry_motion" and (
            not self.semantic_enabled or not self.motion_enabled
        ):
            raise ValueError(
                "semantic_geometry_motion carrier mode requires Semantic and Motion"
            )
        if self.variant == "whole_scene" and self.target_scope != "whole_scene":
            raise ValueError("Whole-scene control must use Whole-scene targets")
        if self.variant != "whole_scene" and self.target_scope != "task_relevant":
            raise ValueError("Only the Whole-scene control may change target scope")


ABLATION_SPECS: dict[AblationVariant, AblationSpec] = {
    "geometry_only": AblationSpec(
        variant="geometry_only",
        display_name="Geometry only",
        target_scope="task_relevant",
        internal_mode="geometry",
        semantic_enabled=False,
        geometry_enabled=True,
        motion_enabled=False,
        action_conditioning=("geometry",),
        lambda_geo=0.05,
        lambda_sem=None,
        lambda_motion=None,
    ),
    "semantic_geometry": AblationSpec(
        variant="semantic_geometry",
        display_name="Semantic + Geometry",
        target_scope="task_relevant",
        internal_mode="semantic_geometry",
        semantic_enabled=True,
        geometry_enabled=True,
        motion_enabled=False,
        action_conditioning=("geometry",),
        lambda_geo=0.05,
        lambda_sem=0.01,
        lambda_motion=None,
    ),
    # The reviewed core has no public Semantic+Motion mode.  The isolated
    # ablation policy uses its S+G+M carrier config, then removes Geometry's
    # parameters and span before any checkpoint load or forward pass.
    "semantic_motion": AblationSpec(
        variant="semantic_motion",
        display_name="Semantic + Motion",
        target_scope="task_relevant",
        internal_mode="semantic_geometry_motion",
        semantic_enabled=True,
        geometry_enabled=False,
        motion_enabled=True,
        action_conditioning=("motion",),
        lambda_geo=None,
        lambda_sem=0.01,
        lambda_motion=0.05,
    ),
    "full": AblationSpec(
        variant="full",
        display_name="Full SGeM-VLA",
        target_scope="task_relevant",
        internal_mode="semantic_geometry_motion",
        semantic_enabled=True,
        geometry_enabled=True,
        motion_enabled=True,
        action_conditioning=("geometry", "motion"),
        lambda_geo=0.05,
        lambda_sem=0.01,
        lambda_motion=0.05,
    ),
    "supervision_only": AblationSpec(
        variant="supervision_only",
        display_name="Supervision-only",
        target_scope="task_relevant",
        internal_mode="semantic_geometry_motion",
        semantic_enabled=True,
        geometry_enabled=True,
        motion_enabled=True,
        action_conditioning=(),
        lambda_geo=0.05,
        lambda_sem=0.01,
        lambda_motion=0.05,
    ),
    "whole_scene": AblationSpec(
        variant="whole_scene",
        display_name="Whole-scene control",
        target_scope="whole_scene",
        internal_mode="semantic_geometry_motion",
        semantic_enabled=True,
        geometry_enabled=True,
        motion_enabled=True,
        action_conditioning=("geometry", "motion"),
        lambda_geo=0.05,
        lambda_sem=0.01,
        lambda_motion=0.05,
    ),
}


def get_ablation_spec(variant: str) -> AblationSpec:
    try:
        return ABLATION_SPECS[variant]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"unsupported RoboCasa ablation: {variant}") from exc
