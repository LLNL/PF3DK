/*
 * rotth.c
 *
 * does rotation of complex array tvar through real angle thetb
 *
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>

#include "light.h"
#include "pf3dbench.h"
#include "util.h"

#include "runparm.h"
#include "pf3dbenchvars.h"

#define tvar3D(a,b,c) tvar[CELTNDX3(a,b,c)]
#define thetb3D(a,b,c) thetb[CELTNDX3(a,b,c)]

void rotth_mult_omp45(int nzl, rcomplex * restrict tvar, real * restrict thetb)
{
#pragma omp target teams num_teams(num_teams) map(to:thetb[0:ngtot]) map(tofrom:tvar[0:ngtot])
  {
    int iz;
    
#ifdef _OPENMP
#pragma omp distribute private(iz)
#endif
    for (iz= 0; iz < nzl; iz++) {
        rotth_omp45(tvar, thetb, iz);
    }
  }
}

void rotth_mult_omp45_pre(int nzl, rcomplex * restrict tvar, real * restrict thetb)
{
    int  ix, iy, iz;

#pragma omp target teams num_teams(num_teams)
    {
#pragma omp distribute private(iz)
      for (iz= 0; iz < nzl; iz++) {

#ifdef _OPENMP
#pragma omp parallel for COLLAPSE(2) private(ix, iy)
#endif
        for (iy=0; iy<nyl; iy++) {
  	  for (ix=0; ix<nxl; ix++) {
            tvar3D(ix,iy,iz)= tvar3D(ix,iy,iz)*( COS(thetb3D(ix,iy,iz))+IREAL*SIN(thetb3D(ix,iy,iz)) );
	  }
        }
      }
    }
  /* No need to bring data back with:
     #pragma omp target update from(tvar[0:ngtot])
     That will happen when the unmap call is made.
  */
}

void rotth_premap(rcomplex * restrict tvar, real * restrict thetb)
{
#pragma omp target enter data map(to:thetb[0:ngtot],tvar[0:ngtot])
}

void rotth_unmap(rcomplex * restrict tvar, real * restrict thetb)
{
#pragma omp target exit data map(from:tvar[0:ngtot]) map(release:thetb[0:ngtot])
}

void rotth_omp45(rcomplex * restrict tvar, real * restrict thetb, int iz)
{
    int  ix, iy;

#ifdef _OPENMP
#pragma omp parallel for COLLAPSE(2) private(ix, iy)
#endif
    for (iy=0; iy<nyl; iy++) {
	for (ix=0; ix<nxl; ix++) {
          tvar3D(ix,iy,iz)= tvar3D(ix,iy,iz)*( COS(thetb3D(ix,iy,iz))+IREAL*SIN(thetb3D(ix,iy,iz)) );
	}
    }
}

void rotth_omp45_pre3D(int nzl, rcomplex * restrict tvar, real * restrict thetb,
                       int izlo, int izhi)
{
  /* #pragma omp target teams num_teams(num_teams)  map(to:thetb[0:ngtot]) map(tofrom:tvar[0:ngtot]) */
#pragma omp target teams num_teams(num_teams) map(to:izlo,izhi)
  {
    int ix, iy, iz;
    
#pragma omp distribute parallel for COLLAPSE(3)
    for (iz= izlo; iz <= izhi; iz++) {
        for (iy=0; iy<nyl; iy++) {
	    for (ix=0; ix<nxl; ix++) {
              tvar3D(ix,iy,iz)= tvar3D(ix,iy,iz)*( COS(thetb3D(ix,iy,iz))+IREAL*SIN(thetb3D(ix,iy,iz)) );
            }
	}
    }
  }
  /* No need to bring data back with:
     #pragma omp target update from(tvar[0:ngtot])
     That will happen when the unmap call is made.
  */
}
