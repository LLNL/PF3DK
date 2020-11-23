#!/bin/csh
#

# WARNING!! Get an interactive prompt before running this script!
#### DO NOT USE!! ###   bsub -nnodes 1 -Is -W 240 -G guests /usr/bin/tcsh
#   lalloc 1 -W 240

setenv CODE "~/repo/git/pf3d-fft-test/${SYS_TYPE}/pf3dtest-mpi"
setenv OMP_STACKSIZE 16M

echo "Running on the Power 9 Volta GPU"
# NOTE - the number of zones per thread is deliberately larger than
# normal host CPU runs to try and get enough work to keep the GPU busy
lrun -N 1 -n 40 $CODE 128 160   80   1 1 -O3
