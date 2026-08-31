#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [[ -z "${UV_BIN}" || ! -x "${UV_BIN}" ]]; then
  echo "uv is not installed. Follow https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
fi

cd "${REPO_ROOT}"
git submodule update --init --recursive
"${UV_BIN}" python install 3.11
if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.11 .venv
fi
GIT_LFS_SKIP_SMUDGE=1 "${UV_BIN}" sync --frozen

TRANSFORMERS_DIR="$(
  "${REPO_ROOT}/.venv/bin/python" -c \
    'from pathlib import Path; import transformers; print(Path(transformers.__file__).parent)'
)"
cp -a "${REPO_ROOT}/src/openpi/models_pytorch/transformers_replace/." "${TRANSFORMERS_DIR}/"

"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import sys

import torch
import transformers

if sys.version_info[:2] != (3, 11):
    raise RuntimeError(f"expected Python 3.11, observed {sys.version}")
if torch.__version__.split("+")[0] != "2.7.1":
    raise RuntimeError(f"expected torch 2.7.1, observed {torch.__version__}")
if transformers.__version__ != "4.53.2":
    raise RuntimeError(
        f"expected transformers 4.53.2, observed {transformers.__version__}"
    )
print(
    {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_visible_on_this_node": torch.cuda.is_available(),
        "visible_gpu_count": torch.cuda.device_count(),
    }
)
PY
