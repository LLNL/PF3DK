**PF3DK**
=======

PF3DK contains kernels extracted from pF3D, A laser-plasma simulation
code developed at LLNL. The kernels are intended for use in evaluating
the effectiveness of compiler optimizations.

Quick Start
-----------

The pF3DK repo contains build and run script for HPC systems at LLNL.
To get started, make a copy of a build and a run script and modify
them to match your system. You can type something like this to run
on an LLNL Broadwell cluster:

```shell
# build a reference code
./build-topaz-mpi-ref.sh

# build an optimized code
./build-topaz-mpi.sh

# allocate a node to run on
salloc -N 1 -n 1 --exclusive

# This writes the results of the reference code to a text file
./run-topaz-mpi-ref.csh

# run the optimized code to measure performance and compare to
# the reference results
./run-topaz.csh
```

The next section briefly describes pF3D and the kernels extracted from it.
If you are just interested in building and running the kernels, jump down
to the Building and Running sections.


Overview of pF3D
-----------

pF3D is used to simulate the interaction between a high intensity laser
and a plasma (ionized gas) in experiments performed at LLNL's National
Ignition Facility. pF3D simulations can consume large amounts of computer
time, so it is important that it is well optimized.

pF3D uses many complex arrays and a few real arrays. These arrays
normally have float precision. Versions of the testsThe pF3DK kernels run
on CPUs and GPUs.

PF3DK includes rotth and couple_waves kernels. These
kernels are loosely based on functions in pF3D, but have been
simplified to more clearly exhibit some compiler issues that
have arisen during the development of pF3D.

The key points about these functions from a compiler point
of view is that they use C99 complex variables, compute
sines and cosines, and have float complex, float and double
variables. The loops are SIMDizable and have OpenMP simd directives
on CPUs. The challenges are for the compiler to recognize that a loop
containing multiple data types and calls to math libraries
is SIMDizable. For bonus points a compiler needs to figure
out the correct SIMD size to use when the CPU supports multiple
vector widths (hint- the goal is to make the sines and cosines
fast). OpenMP 4.5 target offload is used by the GPU versions.

PF3DK also includes some 2D FFT kernels. These kernels
perform 2D FFTs over xy-planes. A 2D FFT is performed
for all xy-planes so the FFT kernel operates on 3D arrays.

pF3D normally runs with xy-planes decomposed into multiple
MPI domains. Performing a 2D FFT is done by "transposing"
uso that all processes have complete rows of the grid. 1D FFTs
are performed on each row using a single process FFT.
The FFTs use the FFTW API or a vendor specific API
that is similar to FFTW (e.g. Nvidia cuFFT). Another transpose
is used to assemble complete columns in each process. A second set
of 1D FFTs is performed, and a final transpose takes the data
back to the original checkerboard decomposition.

The 2D FFTs are implemented using a "toolkit" of functions
that handle data movement. The FFTs can be performed using FFTW
or a vendor optimized FFT library. There are versions that
run in a single process and versions that run on multiple
MPI processes. The use of the toolkit instead of a single
monolithic function makes it easier to check the performance
of the individual pieces.


Building PF3DK
-----------

The kernels have very few external dependencies so they are easy
to build. The compiler flag to enable OpenMP support should be set on
compile and link lines in the build scripts. An MPI wrapper (e.g. mpicc)
for the C compiler is assumed to be available. If it isn't, you will
need to add the appropriate include files to the compile lines and
the appropriate libraries to the load line.

The x86_64 build scripts are set up to use the Intel C compiler and
link with the Intel MKL library to access a tuned FFT. Tuned FFTs
should be used when available.


Running PF3DK
-----------

The kernels run in a few minutes and you may  want to try them
with several different compiler flags in one session. The best
way to do that at LLNL is to allocate a single node and run an
interactive shell on it (use salloc on a SLURM based system).

There is a script that runs a code built with very low optmization
to generate reference values. Another script runs an optimized code
and compares the results to the reference values.

The scripts for "topaz" are intended for 36 core Intel Broadwell nodes.
The scripts for "ansel" are intended for 40 core IBM Power 9 nodes
with 4 Nvidia Tesla GPUs. One pair of scripts checks only CPU
performance. The other checks both CPU and GPU performance.

The build and run scripts rely on the SYS_TYPE variable used
at LLNL. To create your own build and run scripts, start with
the sample scripts and ajdust them to match your system.

License
-----------

PF3DK is distributed under the terms of the BSD-3 license. All new contributions must be made under the BSD-3 license.

See LICENSE-BSD and NOTICE for details.

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-814803
