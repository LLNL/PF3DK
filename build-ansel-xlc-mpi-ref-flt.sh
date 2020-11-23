#!/bin/sh -i
#
# For SYS_TYPE blueos_3_ppc64le_ib
EXE_NAME=pf3dtest-mpi-ref
export EXE_NAME

echo "Using xlc to build for rzansel"
mkdir -p $SYS_TYPE

# issue a make clean so that optimization does not become mixed up
# when switching between reference and optimized builds.
make -f Makefile-mpi clean
make -f Makefile-mpi CODE_NAME=$EXE_NAME  CC="mpixlc -g8 -std=c99 -qsmp=omp -DCOMPLEXMACRO -DUSE_FFTW -DBUILD_REF -DUSE_MPI"  CFLAGS=""  COPTIMIZE="-O1 -qreport" LDFLAGS="-std=c99 -qreport /usr/tcetmp/packages/fftw/fftw-3.3.7/lib/libfftw3.a -lm"
