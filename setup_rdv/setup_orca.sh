#!/bin/bash
export OPAL_PREFIX=/opt/adm/compile/openmpi/4.1.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/adm/compile/openmpi/4.1.1/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/adm/orca/5.0.4/dynamic_openmpi411
export PATH=${PATH}:/opt/adm/orca/5.0.4/dynamic_openmpi411
export PATH=${PATH}:/opt/adm/compile/openmpi/4.1.1/bin
export ORCA=/opt/adm/orca/5.0.4/dynamic_openmpi411

export SCRADIR=$SCRATCH
source /opt/adm/theodore/2.4.0/setpaths.bash
export THEODORE=$THEODIR
