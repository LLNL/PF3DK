#!/bin/csh
#

# WARNING!! Get an interactive prompt before running this script!
#### DO NOT USE!! ###   bsub -nnodes 1 -Is -W 240 -G guests /usr/bin/tcsh
#   lalloc 1 -W 240

setenv CODE "~/repo/git/pf3d-fft-test/${SYS_TYPE}/pf3dtest-gpu-mpi"
setenv OMP_STACKSIZE 16M

echo "Running on the Power 9 Volta GPU"
# NOTE - the number of zones per thread is deliberately larger than
# normal host CPU runs to try and get enough work to keep the GPU busy
ml cuda
# WARNING - this test has 10 OMP threads per MPI process
# and that might cause some differences compared to pure MPI.
lrun -N 1 -n 4 --smpiargs="-gpu" nvprof $CODE 128 192   80   10 1 -O3
# lrun -N 1 -n 4 $CODE 128 192   80   10 1 -O3
# lrun -N 1 -n 4 --smpiargs="-gpu" nvprof -o nvprof-rzansel-%q{OMPI_COMM_WORLD_RANK}.prof $CODE 128 192   80   10 1 -O3
