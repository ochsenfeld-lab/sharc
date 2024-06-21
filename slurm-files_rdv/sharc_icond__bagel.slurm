#!/bin/bash
#SBATCH --job-name=sharc_icond__bagel
#SBATCH --output=sharc_icond__bagel.%j.slurm.out
#SBATCH --error=sharc_icond__bagel.%j.slurm.err
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --partition=freya
#SBATCH --mem=2048
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source /home.io/map/sharc/sharc_3.0.1/sharc/bin/sharcvars.sh
source $SHARC_BASE/setup_rdv/setup_bagel.sh

echo $SLURM_NODELIST > hostname
export SCRATCH=/scratch/$SLURM_JOB_USER/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID
mkdir -p $SCRATCH
sleep 1s

bash run.sh

