#!/usr/bin/env bash
set -euo pipefail

# The common multi-worker launcher invokes the main server module explicitly.
# This process-only adapter replaces that single module name and delegates every
# argument to the real server Python.  Simulator workers are unaffected.
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
repository_root=$(cd "$project_root/../.." && pwd)
openpi_root=${OPENPI_ROOT:-$repository_root}
python_bin=${ROBOCASA24_ABLATION_SERVER_PYTHON:-$openpi_root/.venv/bin/python}

if [[ ${1:-} == -m && ${2:-} == robocasa24_finetune.eval_server ]]; then
  shift 2
  exec "$python_bin" -m robocasa24_finetune.ablations.eval_server "$@"
fi
exec "$python_bin" "$@"
