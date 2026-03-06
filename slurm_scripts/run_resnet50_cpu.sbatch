#!/bin/bash
#SBATCH --job-name=resnet50
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail


# Build accelerate arguments conditionally
ACCELERATE_ARGS="--cpu --num_machines=1 --num_processes=1"
echo "No-GPU mode"

apptainer exec --cleanenv \
  /software/singularity/Informatica/mia-idl-apptainer/mia_idl_2.2.sif \
  bash -lc "
    echo 'Inside container on ' \$(hostname) ' at ' \$(date)
    echo 'COMMAND: accelerate launch ' ${ACCELERATE_ARGS} '/IDL_master/IDL_Master/accelerate_scripts/resnet50-train_cpu.py'
    accelerate launch ${ACCELERATE_ARGS} ${HOME}/IDL_master/IDL_Master/accelerate_scripts/resnet50-train_cpu.py
  "

