#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/policy_aux_gpu_common.sh" "$@" 4
