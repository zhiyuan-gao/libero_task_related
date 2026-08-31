"""Frozen RoboCasa Atomic-24 training contracts."""

from __future__ import annotations

ACTION_DIM = 12
MODEL_ACTION_DIM = 32
STATE_DIM = 16
ACTION_HORIZON = 50
EXECUTION_HORIZON = 25
EPISODES_PER_TASK = 50

CAMERAS = (
    "robot0_agentview_left",
    "robot0_eye_in_hand",
    "robot0_agentview_right",
)

POLICY_VIEW_NAMES = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

TASKS = (
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter",
    "PnPCounterToStove",
    "PnPStoveToCounter",
    "OpenSingleDoor",
    "CloseSingleDoor",
    "OpenDoubleDoor",
    "CloseDoubleDoor",
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "TurnOnStove",
    "TurnOffStove",
    "CoffeeSetupMug",
    "CoffeeServeMug",
    "CoffeePressButton",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
)

GEOMETRY_DIM = 2048
MOTION_DIM = 256

DATASET_REPO_ID = "robocasa24_atomic_hdf5_base50"
