#!/usr/bin/env bash
#SBATCH --job-name=libero40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run inside a Slurm allocation." >&2
  exit 2
fi
if [[ -z "${FOUR_SUITE_HPC_ENV_FILE:-}" || ! -f "${FOUR_SUITE_HPC_ENV_FILE}" ]]; then
  echo "Set FOUR_SUITE_HPC_ENV_FILE to an absolute hpc.env path at sbatch submission." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${FOUR_SUITE_HPC_ENV_FILE}"

if [[ -z "${OPENPI_ROOT:-}" || ! -d "${OPENPI_ROOT}" ]]; then
  echo "OPENPI_ROOT is unavailable after loading ${FOUR_SUITE_HPC_ENV_FILE}." >&2
  exit 2
fi
if [[ ! -x "${FOUR_SUITE_PYTHON:-}" ]]; then
  echo "FOUR_SUITE_PYTHON is unavailable: ${FOUR_SUITE_PYTHON:-unset}" >&2
  exit 2
fi

GPU_COUNT="$("${FOUR_SUITE_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${GPU_COUNT}" != "8" ]]; then
  echo "Expected exactly 8 visible GPUs in the Slurm job, observed ${GPU_COUNT}." >&2
  nvidia-smi || true
  exit 2
fi

echo "Slurm job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} visible_gpus=${GPU_COUNT}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

cd "${OPENPI_ROOT}/experiments/four_suite_joint"
exec srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 ./jobs/run_8gpu.sh "$@"
