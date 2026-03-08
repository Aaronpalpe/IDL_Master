#!/bin/bash
#SBATCH --job-name=resnet50
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Default value if not provided from outside
NUM_PROCESSES=${NUM_PROCESSES:-1}

echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Build accelerate arguments conditionally
ACCELERATE_ARGS="--num_machines=1 --num_processes ${NUM_PROCESSES}"
if [ "${NUM_PROCESSES}" -gt 1 ]; then
  ACCELERATE_ARGS="--multi_gpu ${ACCELERATE_ARGS}"
  echo "Multi-GPU mode enabled"
else
  echo "Single-GPU mode"
fi

nvidia-smi

apptainer exec --nv --cleanenv \
  /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
  bash -lc "
    echo 'Inside container on ' \$(hostname) ' at ' \$(date)
    echo 'COMMAND: accelerate launch ' ${ACCELERATE_ARGS} '/accelerate_scripts/resnet50-train_mgpu.py'
    accelerate launch ${ACCELERATE_ARGS} ${HOME}/accelerate_scripts/resnet50-train_mgpu_stats2.py
  "

