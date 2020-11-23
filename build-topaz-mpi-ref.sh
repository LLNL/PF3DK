#!/bin/bash -i
#
# For SYS_TYPE toss_3_x86_64_ib
EXE_NAME=pf3dtest-mpi-ref
export EXE_NAME

# The reference code will be built using double precision and
# and -O0 optimization. Representative array values will be saved
# for each test.

module load intel/19.0.4
MKLROOT=/usr/tce/packages/mkl/mkl-2019.0
export MKLROOT
echo "MKLROOT is ${MKLROOT}"

mkdir -p $SYS_TYPE

echo "Using icc MPI to build reference code for rztopaz"

# issue a make clean so that optimization does not become mixed up
# when switching between reference and optimized builds.
make -f Makefile-mpi clean
make -f Makefile-mpi CODE_NAME=$EXE_NAME CC="mpicc -g -std=c99 -fno-alias -O0 -qopenmp -DNUM_TEAMS=1 -DUSE_MKL -DUSE_FFTW -DUSE_MPI -DBUILD_REF -DUSE_DOUBLE -I ${MKLROOT}/include" CFLAGS="" COPTIMIZE="-O0"  LDFLAGS="-std=c99 -Wl,--start-group ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_core.a ${MKLROOT}/lib/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm -ldl"
