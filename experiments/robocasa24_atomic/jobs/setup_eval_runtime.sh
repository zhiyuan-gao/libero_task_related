#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repository_root=$(cd "$project_root/../.." && pwd)
openpi_root=${OPENPI_ROOT:-$repository_root}
runtime_root=${ROBOCASA_EVAL_RUNTIME_ROOT:-$project_root/.runtime/eval}
robocasa_root="$runtime_root/robocasa"
robosuite_root="$runtime_root/robosuite"
venv="$runtime_root/.venv"
robocasa_commit=756598a5be52e052339bb2d957426e39015c2afb
robosuite_commit=cb173eb465089b1b4d7038dc8e913f18817f2b0f

mkdir -p "$runtime_root"
if [[ ! -d "$robocasa_root/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/robocasa/robocasa.git "$robocasa_root"
fi
git -C "$robocasa_root" fetch origin "$robocasa_commit"
git -C "$robocasa_root" checkout --detach "$robocasa_commit"

if [[ ! -d "$robosuite_root/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/ARISE-Initiative/robosuite.git "$robosuite_root"
fi
git -C "$robosuite_root" fetch origin "$robosuite_commit"
git -C "$robosuite_root" checkout --detach "$robosuite_commit"

if [[ ! -x "$venv/bin/python" ]]; then
  uv venv --python 3.10 "$venv"
fi
uv pip install --python "$venv/bin/python" \
  numpy==1.23.3 numba==0.56.4 scipy==1.10.1 mujoco==3.2.6 \
  pillow opencv-python-headless pynput termcolor tqdm imageio imageio-ffmpeg \
  h5py lxml hidapi pyyaml pygame qpsolvers quadprog dm-tree msgpack websockets
uv pip install --python "$venv/bin/python" --no-deps -e "$robosuite_root" -e "$robocasa_root"
uv pip install --python "$venv/bin/python" -e "$openpi_root/packages/openpi-client"

echo "RoboCasa evaluation runtime ready: $venv"
echo "RoboCasa commit: $(git -C "$robocasa_root" rev-parse HEAD)"
echo "RoboSuite commit: $(git -C "$robosuite_root" rev-parse HEAD)"
if [[ "${DOWNLOAD_ASSETS:-0}" == 1 ]]; then
  printf 'y\n' | PYTHONPATH="$robocasa_root:$robosuite_root" \
    "$venv/bin/python" -m robocasa.scripts.download_kitchen_assets
else
  echo "Assets were not downloaded. Re-run with DOWNLOAD_ASSETS=1 before simulator smoke/evaluation."
fi
