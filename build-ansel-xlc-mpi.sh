#!/bin/sh -i
#
# For SYS_TYPE blueos_3_ppc64le_ib
EXE_NAME=pf3dtest-mpi
export EXE_NAME

echo "Using xlc to build for rzansel"
mkdir -p $SYS_TYPE

# make -f Makefile-mpi CODE_NAME=$EXE_NAME  CC="mpixlc -g8 -std=c99 -qsmp=omp -DCOMPLEXMACRO -DUSE_FFTW -DUSE_MPI"  CFLAGS=""  COPTIMIZE="-O3 -qtune=pwr9 -qarch=pwr9 -qreport" LDFLAGS="-std=c99 -qreport /usr/tcetmp/packages/fftw/fftw-3.3.7/lib/libfftw3.a -lm"

make -f Makefile-mpi CODE_NAME=$EXE_NAME  CC="mpixlc -g8 -std=c99 -qsmp=omp -DCOMPLEXMACRO -DUSE_FFTW -DUSE_MPI"  CFLAGS=""  COPTIMIZE="-O3 -qtune=pwr9 -qarch=pwr9 -qreport" LDFLAGS="-std=c99 -qreport /usr/tcetmp/packages/fftw/fftw-3.3.7/lib/libfftw3.a -lm"
