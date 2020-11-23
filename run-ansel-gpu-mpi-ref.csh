#!/bin/csh
#

# WARNING!! Get an interactive prompt before running this script!
#### DO NOT USE!! ###   bsub -nnodes 1 -Is -W 240 -G guests /usr/bin/tcsh
#   lalloc 1 -W 240

setenv CODE "./${SYS_TYPE}/pf3dtest-gpu-mpi-ref"
setenv OMP_STACKSIZE 16M

echo "Running on the Power 9 Volta GPU"
# NOTE - the number of zones per thread is deliberately larger than
# normal host CPU runs to try and get enough work to keep the GPU busy
# WARNING - this run builds a reference for 10 OMP threads per MPI process
# and that might cause some differences compared to pure MPI.
# lrun -N 1 -n 4 nvprof $CODE 128 160   80   10 1 ref
lrun -N 1 -n 4 nvprof $CODE 128 160   80   1 1 ref
