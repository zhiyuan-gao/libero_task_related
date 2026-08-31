#!/usr/bin/env bash
set -euo pipefail

# Reuse the frozen simulator/worker/summarizer launcher byte-for-byte.  Only
# policy-server construction is routed to the matching ablation topology.
ablation_jobs=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$ablation_jobs/../.." && pwd)
repository_root=$(cd "$project_root/../.." && pwd)
openpi_root=${OPENPI_ROOT:-$repository_root}

export ROBOCASA24_ABLATION_SERVER_PYTHON=${SERVER_PYTHON:-$openpi_root/.venv/bin/python}
export SERVER_PYTHON=$ablation_jobs/python_module_router.sh
exec bash "$project_root/jobs/eval_checkpoint_multiworker.sh" "$@"
