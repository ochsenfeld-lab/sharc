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

source /opt/adm/sharc/v3.0.1/bin/sharcvars.sh
source /opt/adm/compile/intel_oneapi/setvars.sh
source $SHARC_BASE/setup_rdv/setup_molcas.sh

echo $SLURM_NODELIST > hostname
export SCRATCH=/scratch/$SLURM_JOB_USER/$SLURM_JOB_ID
mkdir -p $SCRATCH
sleep 1s

bash run.sh



