#!/bin/sh -i
#
# For SYS_TYPE blueos_3_ppc64le_ib
EXE_NAME=pf3dtest-gpu-mpi
export EXE_NAME

echo "Using xlc-gpu to build for rzansel"
mkdir -p $SYS_TYPE

make -f Makefile-mpi CODE_NAME=$EXE_NAME  CC="mpixlc -g8 -std=c99 -qsmp=omp -DCOMPLEXMACRO -DUSE_FFTW -DUSE_MPI -DNUM_TEAMS=80 -DOMP45_BUILD -I/usr/tce/packages/cuda/cuda-10.1.243/include"  CFLAGS=""  COPTIMIZE="-O3 -qtune=pwr9 -qarch=pwr9 -qreport"   COMP45="mpixlc -g8 -std=c99 -qsmp=omp -qoffload -DUSE_FFTW -qfullpath -DNUM_TEAMS=80 -Xptxas -v -I/usr/tce/packages/cuda/cuda-10.1.243/include -DCOMPLEXMACRO -DOMP45 -O3"  COMP45_LO="mpixlc -g8 -std=c99 -qsmp=omp -qoffload -DUSE_FFTW -qfullpath -DNUM_TEAMS=80 -Xptxas -v -I/usr/tce/packages/cuda/cuda-10.1.243/include -DCOMPLEXMACRO -DOMP45 -O2"  COMP_cuda="mpixlc -g8 -std=c99 -DUSE_FFTW -qfullpath -Xptxas -v -I/usr/tce/packages/cuda/cuda-10.1.243/include -DCOMPLEXMACRO -DOMP45 -O2"  LDFLAGS="-std=c99 -qreport /usr/tcetmp/packages/fftw/fftw-3.3.7/lib/libfftw3.a -lm -lcufft -lnvToolsExt -L/usr/local/cuda/lib64"  OMP45
