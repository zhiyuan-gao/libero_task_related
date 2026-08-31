from __future__ import annotations

import os
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).with_name("train_semantic_geometry_ab_8gpu.sh")


def _fake_launcher(path: Path, stage: str, exit_code_variable: str) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'echo "{stage}:$1:$EXP_NAME:$FULL_TRAINING_APPROVED" >> "$EVENT_LOG"\n'
        f'exit "${{{exit_code_variable}:-0}}"\n'
    )
    path.chmod(0o755)


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    events = tmp_path / "events.txt"
    a_launcher = tmp_path / "a.sh"
    b_launcher = tmp_path / "b.sh"
    _fake_launcher(a_launcher, "A", "A_EXIT_CODE")
    _fake_launcher(b_launcher, "B", "B_EXIT_CODE")
    environment = {
        **os.environ,
        "FORMAL_AB_TRAINING_APPROVED": "YES",
        "A_EXP_NAME": "a_test",
        "B_EXP_NAME": "b_test",
        "A_LAUNCHER": str(a_launcher),
        "B_LAUNCHER": str(b_launcher),
        "EVENT_LOG": str(events),
    }
    return environment, events


def test_b_starts_only_after_a_completes_successfully(tmp_path: Path) -> None:
    environment, events = _environment(tmp_path)
    result = subprocess.run([SCRIPT], env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert events.read_text().splitlines() == ["A:full:a_test:YES", "B:full:b_test:YES"]
    assert "A completed successfully; starting independent B" in result.stdout


def test_b_does_not_start_if_a_fails(tmp_path: Path) -> None:
    environment, events = _environment(tmp_path)
    environment["A_EXIT_CODE"] = "17"
    result = subprocess.run([SCRIPT], env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 17
    assert events.read_text().splitlines() == ["A:full:a_test:YES"]


def test_formal_training_requires_explicit_confirmation(tmp_path: Path) -> None:
    environment, events = _environment(tmp_path)
    environment.pop("FORMAL_AB_TRAINING_APPROVED")
    result = subprocess.run([SCRIPT], env=environment, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert not events.exists()
    assert "researcher confirmation" in result.stderr
