"""Frozen identities and counts shared by preparation, loading, and validation."""

LIBERO_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
LIBERO_REPO_ID = "physical-intelligence/libero"

FOUR_SUITE_EPISODES = 1_693
FOUR_SUITE_FRAMES = 273_465
FOUR_SUITE_TASKS = 40
FOUR_SUITE_GEOMETRY_VALID = 273_377
FOUR_SUITE_GEOMETRY_INVALID = 88
FOUR_SUITE_MOTION_VALID = 256_401

GEOMETRY_DIM = 2_048
MOTION_DIM = 256

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
SUITE_FRAME_COUNTS = {
    "libero_10": 101_469,
    "libero_goal": 52_042,
    "libero_object": 66_984,
    "libero_spatial": 52_970,
}
SUITE_EPISODE_COUNTS = {
    "libero_10": 379,
    "libero_goal": 428,
    "libero_object": 454,
    "libero_spatial": 432,
}
