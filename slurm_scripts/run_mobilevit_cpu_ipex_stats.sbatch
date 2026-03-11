#!/bin/bash
#SBATCH --job-name=mobilevit-cpu-ipex-stats
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15GB
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# IPEX config enables: cpu=True, ipex=True, mixed_precision=bf16
CONFIG_FILE="${HOME}/accelerate_scripts/config_ipexbase.yaml"

echo "Using config: ${CONFIG_FILE}"

apptainer exec --cleanenv \
  /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
  bash -lc "
    echo 'Inside container on ' \$(hostname) ' at ' \$(date)
    echo 'COMMAND: accelerate launch --config_file ${CONFIG_FILE} /accelerate_scripts/mobilevit-train_cpu_ipex_stats.py'
    accelerate launch --config_file ${CONFIG_FILE} ${HOME}/accelerate_scripts/mobilevit-train_cpu_ipex_stats.py
  "
