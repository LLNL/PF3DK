/* This file contains functions to initialize arrays and data
   structures used in the pf3d kernels.
*/

#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mytypes.h"
#include "runparm.h"
#include "lecuyer.h"
#include "util.h"
#include "time.h"
#include "light.h"
#include "pf3d_fft.h"

#include "pf3dbench.h"
#include "pf3dbenchvars.h"

#include "check.h"

double t0_lo, t2_lo, t0_hi, t2_hi;
double denlw_lo, denlw_hi;

long nbig;
rcomplex *t0_big, *t0_big_sav;

/* tN_new has no guard zones but tvar has one in x and y. */
#define tN_new(a,b,c) tN_new[CELTNDX3(a,b,c)]
#define tvar(a,b,c) tvar[CELTNDX(a,b,c)]
#define tvar_sav(a,b,c) tvar_sav[CELTNDX(a,b,c)]
#define t0_sav(a,b,c) t0_sav[CELTNDX(a,b,c)]
#define t2_sav(a,b,c) t2_sav[CELTNDX(a,b,c)]

void safe_free(void *var)
{
  if(var) free(var);
}

rcomplex *make_wave(double tvar_lo, double tvar_hi)
{
  int ii, i, j, k, ibase, jbase;
  rcomplex *tvar;

  tvar= init_complex_ramp(nxl+1, nyl+1, nzl, tvar_lo, tvar_hi);
  
  return tvar;
}

void free_wave(rcomplex *tvar)
{
  safe_free(tvar);
}

void copy_wave(rcomplex *tvar, rcomplex *tvar_sav)
{
  int ix, iy, iz;

#ifdef _OPENMP
#pragma omp for private(ix, iy, iz)
#endif
  for(iz= 0; iz < nzl; iz++) {
      for (iy=0; iy<nyl; iy++) {
	for (ix=0; ix<nxl; ix++) {
          tvar(ix, iy, iz)= tvar_sav(ix, iy, iz);
        }
      }
  }
}

void copy_to_tN(rcomplex * restrict tvar, rcomplex * restrict tN_new)
{
  int ix, iy, iz;
 
#ifdef _OPENMP
#pragma omp for private(ix, iy, iz)
#endif
    for(iz= 0; iz < nzl; iz++) {
      for (iy=0; iy<nyl; iy++) {
	for (ix=0; ix<nxl; ix++) {
          tN_new(ix, iy, iz)= tvar(ix, iy, iz);
        }
      }
    }
}

void copy_from_tN(rcomplex * restrict tN_new, rcomplex * restrict tvar)
{
  int ix, iy, iz;
 
#ifdef _OPENMP
#pragma omp for private(ix, iy, iz)
#endif
    for(iz= 0; iz < nzl; iz++) {
      for (iy=0; iy<nyl; iy++) {
	for (ix=0; ix<nxl; ix++) {
          tvar(ix, iy, iz)= tN_new(ix, iy, iz);
        }
      }
    }
}

void copy_arr2d(real *arr_old, real *arr_new)
{
  long i;
  
  for(i= 0; i < nplng; i++) {
    arr_new[i]= arr_old[i];
  }
}

void copy_arr3d(real *arr_old, real *arr_new)
{
  long i;
  
  for(i= 0; i < ngtot; i++) {
    arr_new[i]= arr_old[i];
  }
}

void copy_carr3d(rcomplex *arr_old, rcomplex *arr_new)
{
  long i;
  
  for(i= 0; i < ngtot; i++) {
    arr_new[i]= arr_old[i];
  }
}

void do_init(int nxl_in, int nyl_in, int nzl_in, int nthr_in)
{
  int  i, ntot, izz, isign;
  int  idir;             /* +1=forward, -1=backward */
  long nmx;
  
  /* mfh data */
  int ii, jj, kk, ijk, maxfftlen;
  long   nrzone;
  
  int nxp2, nyp2, nzp2, xyplane, xzplane, yzplane, buffer_size;
  
  double t0big, t2big;
  int ix, iy, iz;
  long memtot;

  /* set the maximum number of threads that can be used during this run */
#ifdef _OPENMP
  /* use the same number of threads for all functions */
  num_thr= nthr_in;
  if(num_thr > 0) omp_set_num_threads(num_thr);
  omp_maxthreads= omp_get_max_threads();
#else
  num_thr= 1;
  omp_maxthreads= 1;
#endif
  if(mp_rank == 0) printf("Running with a maximum of %d threads\n", omp_maxthreads);

  nxl= nxl_in;
  nyl= nyl_in;
  nzl= nzl_in;

  t0_lo=   0.25;
  t0_hi=   0.75;
  t2_lo=   3.3e-4;
  t2_hi=   6.5e-4;

  denlw_lo= 1.8e-4;
  denlw_hi= 2.8e-4;
  
  ngrd= 1;
  nxa= nxl+ngrd+ngrd;
  nya= nyl+ngrd+ngrd;
  nza= nzl+ngrd+ngrd;
  nxg0= 1;
  nxg1= nxl;
  nyg0= 1;
  nyg1= nyl;
  nzg0= 1;
  nzg1= nzl;
  /* (2*nx, 2*ny, nz) is the total number of zones across
     all domains. (nxl, nyl, nzl) is the number of zones
     in a single domain.
     The grid is decomposed into (mp_p, mp_q, mp_r) domains.
     The only place that nx, ny are used is in 2D FFTs.
     The caller passes nxl by nyl arrays into an FFT
     and gets nxl by nyl results back. Even code that operates
     in wavenumber space operates on nxl by nyl arrays.
  */
#ifdef USE_MPI
  nx= mp_p*nxl/2;
  ny= mp_q*nyl/2;
  nz= mp_r*nzl-1;
#else
  nx= nxl/2;
  ny= nyl/2;
  nz= nzl-1;
#endif

  isign= 1;
  idir= 1;
  
#if 0
  isubcycle= 50;
  dthyd= 0.208;
  dt= dthyd/isubcycle;
#endif
  dx= 12.57;
  dy= 12.57;
  dz= 20.17;
  lx= nx*dx;
  ly= ny*dy;
  lz= (nz+1)*dz;

  ntot= nxl*nyl*nzl;
  ngtot= nxa*nya*nza;
  nplng= nxa*nya;
  ntheta= 4*2*nplng;

  /* allocate space for random number generator before
     initializng any variables. */
  nmx= ngtot;
  if(nza > nmx) nmx= nza;
  if(nplng > nmx) nmx= nplng;

  nmaxpln= nxa*nya;
  if(nmaxpln < nxa*nza) nmaxpln= nxa*nza;
  if(nmaxpln < nya*nza) nmaxpln= nya*nza;
  
  tmp_dbcom= wmalloc(sizeof(double complex)*ngtot);
  
  nxp2 = nxl + 2 * ngrd+1;
  nyp2 = nyl + 2 * ngrd+1;
  nzp2 = nzl + 2 * ngrd + 1;
  xyplane = nxp2 * nyp2;
  xzplane = nxp2 * nzp2;
  yzplane = nyp2 * nzp2;
  /* In earlier versions of pF3D, FFT message passing only required
     a buffer big enough to hold one xy-plane.
     The port to GPUs required operating on all xy-planes
     simultaneously, so now need to make a 3D temporary.
  */
  buffer_size = nxp2*nyp2*nzp2*sizeof(rcomplex);
  iptmp = wmalloc(buffer_size);
  optmp = wmalloc(buffer_size);
  afftbuf = wmalloc(buffer_size);
  sndbuf = wmalloc(buffer_size/2);
  rcvbuf = wmalloc(buffer_size/2);
  if(!mp_rank) printf("sndbuf is %d bytes\n", buffer_size/2);
  /* initialize GPU buffer arrays for use in 2D FFTs */
#ifdef OMP45_BUILD
  init_gpubuf(ngtot);
#endif
  
  /* thetb is planar in pF3D. Make it 3D here to enable more parallelism.
  */
  thetb= init_real(ngtot, 0.0, 0.523599);
  
  /* make planes with guard zones */
  theta= wmalloc(sizeof(real)*4*2*nplng);
  
  t0= make_wave(t0_lo, t0_hi);
  t0_sav= make_wave(t0_lo, t0_hi);
  t2= make_wave(t2_lo, t2_hi);
  t2_sav= make_wave(t2_lo, t2_hi);
  tN_new= make_wave(t0_lo, t0_hi);
  
  /* save the initial state of the light waves */
  copy_wave(t0_sav, t0);
  copy_wave(t2_sav, t2);

  denlw= make_wave(denlw_lo, denlw_hi);
 
  if(mp_rank == 0) puts("variable initialization complete\n");
  if(mp_rank == 0) printf("total bytes allocated= %e\n", 1.0*totalloc);
}

void do_cleanup(void)
{
  safe_free(theta);

  if(t0) free_wave(t0);
  t0= 0;
  if(t2) free_wave(t2);
  t2= 0;
  if(t0_sav) free_wave(t0_sav);
  t0_sav= 0;
  if(t2_sav) free_wave(t2_sav);
  t2_sav= 0;
  if(tN_new) free_wave(tN_new);
  tN_new= 0;
}
