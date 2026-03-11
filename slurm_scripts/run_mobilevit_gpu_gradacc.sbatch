#!/bin/bash
#SBATCH --job-name=mobilevit
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Build accelerate arguments for single GPU with gradient accumulation
ACCELERATE_ARGS="--num_machines=1 --num_processes 1"
echo "Single-GPU mode with gradient accumulation"

nvidia-smi

apptainer exec --nv --cleanenv \
  /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
  bash -lc "
    echo 'Inside container on ' \$(hostname) ' at ' \$(date)
    echo 'COMMAND: accelerate launch ' ${ACCELERATE_ARGS} '/accelerate_scripts/mobilevit-train_gpu_gradacc.py'
    accelerate launch ${ACCELERATE_ARGS} ${HOME}/accelerate_scripts/mobilevit-train_gpu_gradacc.py
  "
