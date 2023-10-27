#!/bin/bash
source /home.io/map/sharc/sharc-3.0.1//sharc/bin/sharcvars.sh
source /home.io/map/miniconda3/etc/profile.d/conda.sh
conda activate pysharc_3.0
source /opt/adm/compile/intel_oneapi/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home.io/map/miniconda3/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home.io/map/miniconda3/envs/pysharc_3.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/adm/compile/fftw3/3.3.10_gnu/
export HOSTNAME=`hostname`

export PYTHONPATH=/home.io/map/sharc/sharc-3.0.1/pysharc:/home.io/map/sharc/sharc-3.0.1/pysharc/sharc/:$PYTHONPATH
