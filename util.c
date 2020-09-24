/* This file contains functions for memory initialization and
   allocation and other helper functions.
*/
#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mytypes.h"
#include "util.h"
#include "lecuyer.h"
#include "pf3dbench.h"

#ifdef OMP45_BUILD
#include <cuda_runtime.h>
#endif

#include "runparm.h"
#include "pf3dbenchvars.h"

extern void *base_malloc(size_t nbytes);
extern void *um_malloc(size_t nbytes);

extern int     ngrd;             /* number of guard cells */
extern int     nxa, nya, nza;    /* allocated box dimensions */

extern long nthreads;
int nthr_reported= 0;

extern char RUN_TYPE[100];

static int real_report=0;
static int complex_report=0;

long errcnt= 0;
size_t totalloc= 0;
double *dtmp= 0;
long dtmp_size= 0;
double tpi= 6.28318530717959;

real *tmp_ramp=0;

int nthr_kern;

void init_random(int flag)
{
  if(flag) {
    use_random= 1;
    if(mp_rank == 0 || mp_rank == 1) puts("Will initialize using random numbers");
  } else {
    use_random= 0;
    if(mp_rank == 0 || mp_rank == 1) puts("Will initialize using a linear span of the range");
  }
}

void parse_args(int argc, char **argv)
{
  /* Set the size of the domain, the number of threads, etc.
     using command line arguments, if supplied. */
  nxl= 512;
  nyl= 384;
  nzl= 24;
  num_thr= 4;
  strcpy(RUN_TYPE, "");
  /* Command line arguments take precedence in setting nx, ny, nz */
  if(argc > 1) {
    /* have at least nxl */
    sscanf(argv[1], "%d", &nxl);
    if(argc > 2) {
      /* have nyl */
      sscanf(argv[2], "%d", &nyl);
      if(argc > 3) {
        /* have nzl */
         sscanf(argv[3], "%d", &nzl);
        if(argc > 4) {
          /* have numthread */
          sscanf(argv[4], "%d", &num_thr);
          if(mp_rank == 0) printf("requested %d OMP threads\n", num_thr);
          if(argc > 5) {
              /* have flag for random initialization */
              sscanf(argv[5], "%d", &use_random);
            if(argc > 6) {
              /* have flag for random initialization */
              sscanf(argv[6], "%s", RUN_TYPE);
            }
          }
        }
      }
    }
  }
  if(mp_rank == 0) printf("The RUN_TYPE is %s\n", RUN_TYPE);
  if(use_random) {
    if(mp_rank == 0) printf("Will initialize using random numbers\n");
  } else {
    if(mp_rank == 0) printf("Will initialize using strided values\n");
  }
}

void c_print(rcomplex v)
{
  sprintf(cbuf, "%e + I*%e", creal(v), cimag(v));
}


long cache_adjust(long size)
{
  /* adjust size upward (if necessary) so that it is an integral
     multiple of a cache line */
  long newsize;

  newsize= (size+CACHE_LINE-1)/CACHE_LINE;
  newsize *= CACHE_LINE;
  return newsize;
}

void *base_malloc(size_t nbytes)
{
  void *ptr;
  
  totalloc += nbytes;
  /* printf("Total bytes allocated so far %ld\n", totalloc); */
  ptr= malloc(nbytes);
  return ptr;
}

void *um_malloc(size_t nbytes)
{
  void *ptr;
  
  totalloc += nbytes;
  /* printf("Total bytes allocated so far %ld\n", totalloc); */
#ifdef OMP45_BUILD
  cudaMallocManaged(&ptr, nbytes, 0);
#else
  ptr= malloc(nbytes);
#endif
  return ptr;
}

/* this function randomly initializes an integer variable with 
   one a number from lo to hi, inclusive.*/
int *init_int(long n, int lo, int hi)
{
  int ii;
  int *var;
  real delta;
  real rlo, rhi, rvar;

  var= (int *)wmalloc(sizeof(int)*n);
  if(dtmp_size < sizeof(double)*n) {
    if(dtmp) free(dtmp);
    dtmp_size= sizeof(double)*n;
    dtmp= (double *)wmalloc(dtmp_size);
  }
  le_nrandom(0, n, dtmp);
  rlo= lo-0.5;
  rhi= hi+0.5;
#ifdef _OPENMP
#pragma omp for private(ii)
#endif
  for(ii= 0; ii < n; ii++) {
    rvar= rlo+(rhi-rlo+1)*dtmp[ii]+0.5;
    var[ii]= (int)rvar;
    if(var[ii] > hi) var[ii]= hi;
    if(var[ii] < lo) var[ii]= lo;
  }
  return var;
}

/* This function creates a real variable, but does not
   initialize it. */
real *make_real(long n)
{
  real *var;

  var= (real *)wmalloc(sizeof(real)*n);
  return var;
}

void linear_real(int num, real *var, real low, real high)
{
  int i;
  real delta;

  delta= (high-low)/(num-1.0);
  for(i= 0; i < num; i++) {
    var[i]= low+delta*i;
  }
}

void linear_rcomp(int num, rcomplex *var, rcomplex low, rcomplex high)
{
  int i;
  rcomplex delta;

  delta= (high-low)/(num-1.0);
  for(i= 0; i < num; i++) {
    var[i]= low+delta*i;
  }
}

/* this function initializes a variable with uniformly distributed
   random values between specified lo and hi values.*/
real *init_real(long n, double lo, double hi)
{
  int ii;
  real *var;

  var= (real *)wmalloc(sizeof(real)*n);
  real_set(n, lo, hi, var);
  
  return var;
}

/* this function initializes a variable with uniformly distributed
   random values between specified lo and hi values.*/
real *init_real_ramp(int numx, int numy, int numz, double lo, double hi)
{
  int ii;
  real *var;

  /* Assume init_real_ramp is only used for 3D arrays and allocate 
     space for guard zones on all sides (often not needed) */
  var= (real *)wmalloc(sizeof(real)*ngtot);
  real_set_ramp(numx, numy, numz, lo, hi, var);
  
  return var;
}

void real_set(long n, double lo, double hi, real *var)
{
  int ii;
  real delta;

  if(use_random) {
    if(dtmp_size < sizeof(double)*n) {
      if(dtmp) free(dtmp);
      dtmp_size= sizeof(double)*n;
      dtmp= (double *)wmalloc(dtmp_size);
    }
    le_nrandom(0, n, dtmp);
#ifdef _OPENMP
#pragma omp for private(ii)
#endif
    for(ii= 0; ii < n; ii++) {
      var[ii]= lo+(hi-lo)*dtmp[ii];
    }
  } else {
    delta= (hi-lo)/n;
    if(real_report) {
      puts("using scaled complex init");
      puts("using strided initialization");
      real_report= 0;
    }
#ifdef _OPENMP
#pragma omp for private(ii)
#endif
    for(ii= 0; ii < n; ii++) {
      var[ii]= lo+delta*ii;
    }
  }
}

void real_set_ramp(int numx, int numy, int numz, double lo,
                   double hi, real *var)
{
  int ii, jj, kk, tmp_size, ndx;
  real modulus, phase, ampl, rampval;

  if(!tmp_ramp) {
      tmp_size= sizeof(real)*nzl;
      tmp_ramp= (real *)wmalloc(tmp_size);
  }
  /* create a 1D ramp from lo to hi */
  linear_real(numz, tmp_ramp, lo, hi);
  ampl= 0.1;
  /* The base value for var(ii,jj,kk) is tmp_ramp(kk).
     dtmp contains random numbers between zero and one.
     Use them to add fluctuations to the "ramp value". */
#ifdef _OPENMP
#pragma omp for private(ii)
#endif
  for(kk= 0; kk < numz; kk++) {
    rampval= tmp_ramp[kk];
    for(jj= 0; jj < numy; jj++) {
      for(ii= 0; ii < numx; ii++) {
        ndx= ii+numx*jj+numx*numy*kk;
        var[ndx]= (1.0 + ampl*(1.0-2.0*le_random(0)) )*rampval;
      }
    }
  }
}

rcomplex *init_complex(long n, double lo, double hi)
{
  int ii;
  rcomplex *var;
  real *dat;
  
  dat= init_real(2*n, 0.0, 1.0);
  var= (rcomplex *)dat;
  complex_set(n, lo, hi, var);
  
  return var;
}

rcomplex *init_complex_ramp(int numx, int numy, int numz, double lo, double hi)
{
  int ii;
  rcomplex *var;
  real *dat;

  /* Assume init_complex_ramp is only used for 3D arrays and allow
     space for guard zones on all sides (often not needed) */
  dat= init_real(2*ngtot, 0.0, 1.0);
  var= (rcomplex *)dat;
  complex_set_ramp(numx, numy, numz, lo, hi, var);
  
  return var;
}

void complex_set(long n, double lo, double hi, rcomplex *var)
{
  int ii;
  real modulus, phase;

#ifdef USE_MODULUS
  /* Treat a pair of values as the modulus and phase for a
     complex number. */
#ifdef _OPENMP
#pragma omp for private(ii)
#endif
  for(ii= 0; ii < n; ii++) {
    modulus= CREAL(var[ii]);
    phase= tpi*CIMAG(var[ii]);
    var[ii]= (lo+(hi-lo)*modulus)*(COS(phase)+IREAL*SIN(phase));
  }
#else
  if(complex_report) {
    puts("using scaled complex init");
    complex_report= 0;
  }
  /* Treat a pair of values as a point in a unit complex square.
     Scale to the requested range. */
#ifdef _OPENMP
#pragma omp for private(ii)
#endif
  for(ii= 0; ii < n; ii++) {
    var[ii]= lo+(hi-lo)*var[ii];
  }
#endif
}

void complex_set_ramp(int numx, int numy, int numz, double lo,
                      double hi, rcomplex *var)
{
  int ii, jj, kk, tmp_size, ndx;
  real modulus, phase, ampl, rampval;

  if(!tmp_ramp) {
      tmp_size= sizeof(real)*nzl;
      tmp_ramp= (real *)wmalloc(tmp_size);
  }
  /* create a 1D ramp from lo to hi */
  linear_real(numz, tmp_ramp, lo, hi);
  ampl= 0.02;
  /* On entry, var contains random numbers between zero and one.
     Treat the first one as a multiplier for the "ramp value"
     in tmp and the second as a random angle. */
#ifdef _OPENMP
#pragma omp for private(ii)
#endif
  for(kk= 0; kk < numz; kk++) {
    rampval= tmp_ramp[kk];
    for(jj= 0; jj < numy; jj++) {
      for(ii= 0; ii < numx; ii++) {
        ndx= ii+numx*jj+numx*numy*kk;
        modulus= ( 1.0+ampl*(1.0-2.0*le_random(0)) )*rampval;
        phase= tpi*le_random(0);
        var[ndx]= modulus*(COS(phase)+IREAL*SIN(phase));
      }
    }
  }
}

void kernel_get_time(int nthr, double *tstart, double *tstop)
{
  int i, nhi;

  nhi= nthr;
  if(nhi > MAX_NTHREAD) nhi= MAX_NTHREAD;
  for(i= 0; i < nhi; i++) {
    tstart[i]= startWallTime[i];
    tstop[i]= endWallTime[i];
  }
}

int kernel_get_nthr(void)
{
  /* printf("nthr_kern=%d\n", nthr_kern); */
  return nthr_kern;
}

void start_omp_time(void)
{
  int threadID;

#ifdef _OPENMP
#pragma omp parallel shared(startWallTime, nthr_kern) private(threadID)
  {
    threadID = omp_get_thread_num();   
#pragma omp single  
    {
      nthr_kern = omp_get_num_threads();
    }
    if(!nthr_reported && threadID == 0) {
      if(mp_rank == 0) printf("number of threads per MPI process is %d\n", nthr_kern);
      nthr_reported= nthr_kern;
    }
    startWallTime[threadID] = omp_get_wtime();
  }
#else
  threadID = 0;
  startWallTime[threadID] = 0.0;
#endif
}

void stop_omp_time(void)
{
  int threadID;
  
#ifdef _OPENMP
#pragma omp parallel shared(startWallTime,endWallTime,totalWallTime) private(threadID)
  {
    threadID = omp_get_thread_num();   
    endWallTime[threadID] = omp_get_wtime();
    totalWallTime[threadID] = endWallTime[threadID] - startWallTime[threadID];
#ifdef FULLDBG
    printf("thread %d took time %e\n", threadID, totalWallTime[threadID]);
#endif
  }
#else
  threadID = 0;
  endWallTime[threadID] = 0.0;
  totalWallTime[threadID] = 0.0;
#endif
}
