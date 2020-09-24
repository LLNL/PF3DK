#include "mytypes.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include "load_data.h"

/* Put extern declarations in an OpenMP "declare target" directive */
#define OMP45_DECLARE
#define __parm_init__

#include "runparm.h"
#include "pf3dbench.h"

#include "pf3dbenchvars.h"

/* tN_new has no guard zones but tvar has one in x and y. */
#define tN_new(a,b,c) tN_new[CELTNDX3(a,b,c)]
#define tvar(a,b,c) tvar[CELTNDX(a,b,c)]


void load_scalars_omp45(int nx_in, int ny_in, int nxl_in,
                        int nyl_in, int nzl_in, int nxa_in, int nya_in,
                        int nza_in, int ngrd_in, long nplng_in, long ngtot_in, 
                        int num_teams_in, long ntheta_in,
                        real clight_in, real csound_in, 
                        real dt_in, real dx_in, real dy_in, real dz_in,
                        real timunit_in,
                        int mp_rank_in, int mp_size_in, int mp_p_in, int mp_q_in,
                        int mp_r_in, int mp_myp_in, int mp_myq_in, int mp_myr_in)
{
  /* Launch one team and have the lead thread of that team do everything */
#pragma omp target teams num_teams(1)
  {
    nx= nx_in; ny= ny_in; nxl= nxl_in; nyl= nyl_in; nzl= nzl_in;
    nxa= nxa_in; nya= nya_in; nza= nza_in; ngrd= ngrd_in; nplng= nplng_in;
    ngtot= ngtot_in; ntheta= ntheta_in;
    num_teams= num_teams_in;
    clight= clight_in; csound= csound_in;
    dt= dt_in; dx= dx_in; dy= dy_in; dz= dz_in;
    timunit= timunit_in;
    mp_rank= mp_rank_in;
    mp_size= mp_size_in;
    mp_p= mp_p_in;
    mp_q= mp_q_in;
    mp_r= mp_r_in;
    mp_myp= mp_myp_in;
    mp_myq= mp_myq_in;
    mp_myr= mp_myr_in;
    if(mp_rank == 0) printf("nxl=%d, nyl=%d, nzl=%d\n", nxl, nyl, nzl);
  }
  if(mp_rank == 0) puts("scalars have been initialized");
}

void load_arrays_omp45(void)
{
  #pragma omp target enter data map(to:theta[0:ntheta])
 
  #pragma omp target enter data map(to:cbuf[0:NCBUF])

  if(mp_rank == 0) puts("arrays have been initialized");
}

void copy_tN_pre(rcomplex * restrict tN, rcomplex * restrict tN_new)
{
  int ii;
 
#pragma omp distribute parallel for
  for(ii= 0; ii < ngtot; ii++) {
    tN_new[ii]= tN[ii];
  }
}

void copy_to_tN_omp45(rcomplex * restrict tvar, rcomplex * restrict tN_new)
{
  int ix, iy, iz;
 
#ifdef _OPENMP
#pragma omp distribute parallel for private(ix, iy, iz)
#endif
    for(iz= 0; iz < nzl; iz++) {
      for (iy=0; iy<nyl; iy++) {
	for (ix=0; ix<nxl; ix++) {
          tN_new(ix, iy, iz)= tvar(ix, iy, iz);
        }
      }
    }
}

void copy_from_tN_omp45(rcomplex * restrict tN_new, rcomplex * restrict tvar)
{
  int ix, iy, iz;
 
#ifdef _OPENMP
#pragma omp distribute parallel for private(ix, iy, iz)
#endif
    for(iz= 0; iz < nzl; iz++) {
      for (iy=0; iy<nyl; iy++) {
	for (ix=0; ix<nxl; ix++) {
          tvar(ix, iy, iz)= tN_new(ix, iy, iz);
        }
      }
    }
}
