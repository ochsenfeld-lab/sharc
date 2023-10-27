#!/bin/bash
source /home.io/map/sharc/sharc-3.0.1/sharc/bin/sharcvars.sh
source /home.io/map/miniconda3/etc/profile.d/conda.sh
conda activate pysharc_3.0
source /opt/adm/compile/intel_oneapi/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home.io/map/miniconda3/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home.io/map/miniconda3/envs/pysharc_3.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/adm/compile/fftw3/3.3.10_gnu/
export HOSTNAME=`hostname`

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/adm/bagel/rev-210120/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/adm/compile/boost/intelmpi_1.69.0/lib
export BAGEL=/opt/adm/bagel/rev-210120/
export PYQUANTE=/opt/adm/pyquante/1.6.5
