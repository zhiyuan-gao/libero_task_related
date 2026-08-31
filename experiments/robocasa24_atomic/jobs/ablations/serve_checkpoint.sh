#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
repository_root=$(cd "$project_root/../.." && pwd)
openpi_root=${OPENPI_ROOT:-$repository_root}
python_bin=${SERVER_PYTHON:-$openpi_root/.venv/bin/python}

export PYTHONPATH="$project_root/src:$openpi_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m robocasa24_finetune.ablations.eval_server "$@"
