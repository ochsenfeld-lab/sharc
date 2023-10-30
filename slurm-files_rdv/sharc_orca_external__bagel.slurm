#!/bin/bash
#SBATCH --job-name=sharc_orca_external__bagel
#SBATCH --output=sharc_orca_external__bagel.%j.slurm.out
#SBATCH --error=sharc_orca_external__bagel.%j.slurm.err
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --partition=baal
#SBATCH --mem=2048
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source /home.io/map/sharc/sharc_3.0.1/sharc/setup_rdv/setup_sharc.sh
source /home.io/map/sharc/sharc_3.0.1/sharc/setup_rdv/setup_bagel.sh
source /home.io/map/sharc/sharc_3.0.1/sharc/setup_rdv/setup_orca.sh

echo $SLURM_NODELIST > hostname
export SCRATCH=/scratch/$SLURM_JOB_USER/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID
mkdir -p $SCRATCH
sleep 1s

bash run_EXTORCA.sh

