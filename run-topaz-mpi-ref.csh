#!/bin/csh
#

# To run on a broadwell node:
#   salloc -N 1 -n 36 -t 4:00:00 --exclusive

setenv CODE "~/repo/git/pf3d-fft-test/${SYS_TYPE}/pf3dtest-mpi-ref"
setenv OMP_STACKSIZE 16M

echo "Creating reference output on the Intel Broadwell"
# NOTE - the number of zones per thread is deliberately larger than
# normal host CPU runs to mimic what would be used on a GPU
# NOTE - the intent is that the product of the number of MPI processes
# and the number of OMP threads per process should equal the number
# of cores (36 for a CTS-1 Broadwell)
srun -i0 -u -N 1 -n 36 $CODE 128 192   80   1 1 -ref-O0
