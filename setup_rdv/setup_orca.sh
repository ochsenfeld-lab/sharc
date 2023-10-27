#!/bin/bash
source /home.io/map/sharc/sharc-3.0.1/bin/sharcvars.sh
source /home.io/map/miniconda3/etc/profile.d/conda.sh
conda activate pysharc_3.0
source /opt/adm/compile/intel_oneapi/setvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home.io/map/miniconda3/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home.io/map/miniconda3/envs/pysharc_3.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/adm/compile/fftw3/3.3.10_gnu/
export HOSTNAME=`hostname`

export OPAL_PREFIX=/opt/adm/compile/openmpi/4.1.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/adm/compile/openmpi/4.1.1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/adm/orca/5.0.4/dynamic_openmpi411
export PATH=${PATH}:/opt/adm/orca/5.0.4/dynamic_openmpi411
export PATH=${PATH}:/opt/adm/compile/openmpi/4.1.1/bin
export ORCA=/opt/adm/orca/5.0.4/dynamic_openmpi411

export SCRADIR=/home.io/map/sharc/sharc-3.0.1/TEST/ORCA/WORK/
source /opt/adm/theodore/2.4.0/setpaths.bash
export THEODORE=$THEODIR
