#!/bin/csh
#

# To run on a broadwell node:
#   salloc -N 1 -n 36 -t 4:00:00 --exclusive
#   salloc -N 1 -n 36 -t 4:00:00 -p mi60 --exclusive # 24 core EPYC, 2 socket?
#   salloc -N 1 -n 36 -t 4:00:00 -p cl --exclusive  # cascade lake 24 core, 2 socket?
#   salloc -N 1 -n 36 -t 4:00:00 -p skylake --exclusive  # skylake 20 core, 2 socket
#   salloc -N 1 -n 36 -t 4:00:00 -p p100 --exclusive  # Xeon 18 core, 2 socket

setenv CODE "~/repo/git/pf3d-fft-test/${SYS_TYPE}/pf3dtest-mpi"
setenv OMP_STACKSIZE 16M

echo "Running on the Intel Broadwell"
# NOTE - the intent is that the product of the number of MPI processes
# and the number of OMP threads per process should equal the number
# of cores (36 for a CTS-1 Broadwell)
srun -i0 -u -N 1 -n 36 $CODE 128 160   80   1 1 -O2
