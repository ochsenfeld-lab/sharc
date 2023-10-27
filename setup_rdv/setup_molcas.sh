#!/bin/bash
source /home.io/map/sharc/sharc-3.0.1/bin/sharcvars.sh
source /home.io/map/miniconda3/etc/profile.d/conda.sh
conda activate pysharc_3.0
source /opt/adm/compile/intel_oneapi/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home.io/map/miniconda3/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home.io/map/miniconda3/envs/pysharc_3.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/adm/compile/fftw3/3.3.10_gnu/
export HOSTNAME=`hostname`

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/adm/compile/hdf5/1.10.6/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/adm/compile/armadillo/11.4.1/lib
export PATH=${PATH}:/opt/adm/OpenMolcas/v23.02/bin
export PATH=${PATH}:/opt/adm/OpenMolcas/v23.02
export MOLCAS=/opt/adm/OpenMolcas/v23.02

