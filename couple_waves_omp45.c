/*
 * couple_waves_omp45.c
 *
 * Couple 3 waves contained in C99 complex arrays.
 *
 * OpenMP 4.5 version intended for use with Nvidia boards and similar devices.
 *
 */

#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mytypes.h"

#include "light.h"
#include "pf3dbench.h"
#include "util.h"

#include "runparm.h"
#include "pf3dbenchvars.h"

void couple_omp45(rcomplex * restrict t0, rcomplex * restrict t2, 
                   rcomplex * restrict denp)
{
  start_omp_time();

#pragma omp target map(to:denp[0:ngtot]) map(tofrom:t0[0:ngtot],t2[0:ngtot])
#pragma omp teams num_teams(num_teams)
  {
    real  c20, cslamt, snlamt, r_zlam, r, fratio;
    real  r_fratio, cratio, zac2;
    double  zlam, c2re, c2im;
    rcomplex a0t, a2t, c2, z3;
    int     ix, iy, iz, myteam;
    long    it0;
        
    cratio= 1.0e3;
    fratio = SQRT(0.9);
    r_fratio = ONE/fratio;
    c20 = 0.25 * cratio * r_fratio;  

    myteam = omp_get_team_num();
  
    /* the static directive is to encourage coalesced loads */
#ifdef _OPENMP
#pragma omp distribute parallel for COLLAPSE(3) schedule(static,1) private(c2, a0t, a2t, zlam, r_zlam, snlamt, cslamt, r, z3, ix, iy, iz, it0, zac2, c2re, c2im)
#endif
    for (iz= 0; iz < nzl; iz++) {
      for (iy=0; iy<nyl; iy++) {
        for (ix=0; ix<nxl; ix++) {
          it0= CELTNDX(ix,iy,iz);
          c2 = c20 * denp[it0];
	  c2re = CREAL(c2);   c2im = CIMAG(c2);
	  /* compute lamda = sqrt(|c2|^2) using doubles
             to avoid underflow. */
          zlam = c2re*c2re + c2im*c2im + 1.0e-34;
	  zlam = sqrt(zlam);
	  snlamt = SIN(zlam * dt * HALF);
	  cslamt = COS(zlam * dt * HALF);

          a0t = t0[it0];
	  a2t = t2[it0] * fratio;
	  /* normalize c2 */
	  r_zlam= ONE/(real)zlam;
	  c2 *= r_zlam;
          /* compute the square of c2 after scaling */
	  zac2 = zabs2(c2);
	       
	  /* compute new A0 */
	  z3 = c2 * a2t * snlamt ;
	  t0[it0] = a0t * cslamt - IREAL * z3;
	       
	  /* compute new A2  */
	  r = zac2 * cslamt; 
	  z3 = CONJ(c2) * a0t * snlamt;
	  t2[it0] = ( a2t * r - IREAL * z3 ) * r_fratio;
        } /* end x for-loop */
      } /* end y for-loop */
    } /* end of distribute z loop */
  } /* end of OMP target */
  stop_omp_time();
}

void couple_premap(rcomplex * restrict t0, rcomplex * restrict t2,
                   rcomplex * restrict denp)
{

#pragma omp target enter data map(to:denp[0:ngtot],t0[0:ngtot],t2[0:ngtot])
}

void couple_unmap(rcomplex * restrict t0, rcomplex * restrict t2,
                   rcomplex * restrict denp)
{

#pragma omp target exit data map(release:denp[0:ngtot]) map(from:t0[0:ngtot],t2[0:ngtot])
}

void couple_omp45_pre(rcomplex * restrict t0, rcomplex * restrict t2,
                   rcomplex * restrict denp)
{
  
  start_omp_time();

#pragma omp target teams num_teams(num_teams)
  {
    real  c20, cslamt, snlamt, r_zlam, r, fratio;
    real  r_fratio, cratio, zac2;
    double  zlam, c2re, c2im;
    rcomplex a0t, a2t, c2, z3, z4;
    int     ix, iy, iz, myteam;
    long    it0;
        
    cratio= 1.0e3;
    fratio = SQRT(0.9);
    r_fratio = ONE/fratio;
    c20 = 0.25 * cratio * r_fratio;  

    myteam = omp_get_team_num();
  
    /* the static directive is to encourage coalesced loads */
#ifdef _OPENMP
#pragma omp distribute parallel for COLLAPSE(3) schedule(static,1) private(c2, a0t, a2t, zlam, r_zlam, snlamt, cslamt, r, z3, ix, iy, iz, it0, zac2, c2re, c2im)
#endif
    for (iz= 0; iz < nzl; iz++) {
      for (iy=0; iy<nyl; iy++) {
        for (ix=0; ix<nxl; ix++) {
          it0= CELTNDX(ix,iy,iz);
          c2 = c20 * denp[it0];
	  c2re = CREAL(c2);   c2im = CIMAG(c2);
	  /* compute lamda = sqrt(|c2|^2) using doubles
             to avoid underflow. */
          zlam = c2re*c2re + c2im*c2im + 1.0e-34;
	  zlam = sqrt(zlam);
	  snlamt = SIN(zlam * dt * HALF);
	  cslamt = COS(zlam * dt * HALF);

          a0t = t0[it0];
	  a2t = t2[it0] * fratio;
	  /* normalize c2 */
	  r_zlam= ONE/(real)zlam;
	  c2 *= r_zlam;
          /* compute the square of c2 after scaling */
	  zac2 = zabs2(c2);
	       
	  /* compute new A0 */
	  z3 = c2 * a2t * snlamt ;
	  t0[it0] = a0t * cslamt - IREAL * z3;
	       
	  /* compute new A2  */
	  r = zac2 * cslamt; 
	  z3 = CONJ(c2) * a0t * snlamt;
	  t2[it0] = ( a2t * r - IREAL * z3 ) * r_fratio;
        } /* end x for-loop */
      } /* end y for-loop */
    } /* end of distribute z loop */
  } /* end of OMP target */
  stop_omp_time();
}



void couple_omp45_pre_simd(rcomplex * restrict t0, rcomplex * restrict t2,
                   rcomplex * restrict denp)
{
  
  start_omp_time();

#pragma omp target teams num_teams(num_teams)
  {
    real  c20, cslamt, snlamt, r_zlam, r, fratio;
    real  r_fratio, cratio, zac2;
    double  zlam, c2re, c2im;
    rcomplex a0t, a2t, c2, z3;
    int     ix, iy, iz, myteam;
    long    it0;
        
    cratio= 1.0e3;
    fratio = SQRT(0.9);
    r_fratio = ONE/fratio;
    c20 = 0.25 * cratio * r_fratio;  

    myteam = omp_get_team_num();
  
#ifdef _OPENMP
#pragma omp distribute private(iz)
#endif
    for (iz= 0; iz < nzl; iz++) {

    /* the static directive is to encourage coalesced loads */
#pragma omp parallel for simd COLLAPSE(2) schedule(static,1) private(c2, a0t, a2t, zlam, r_zlam, snlamt, cslamt, r, z3, it0, zac2, c2re, c2im)
      for (iy=0; iy<nyl; iy++) {
        for (ix=0; ix<nxl; ix++) {
          it0= CELTNDX(ix,iy,iz);
          c2 = c20 * denlw[it0];
	  c2re = CREAL(c2);   c2im = CIMAG(c2);
	  /* compute lamda = sqrt(|c2|^2) using doubles
             to avoid underflow. */
          zlam = c2re*c2re + c2im*c2im + 1.0e-34;
	  zlam = sqrt(zlam);
	  snlamt = SIN(zlam * dt * HALF);
	  cslamt = COS(zlam * dt * HALF);

          a0t = t0[it0];
	  a2t = t2[it0] * fratio;
	  /* normalize c2 */
	  r_zlam= ONE/(real)zlam;
	  c2 *= r_zlam;
          /* compute the square of c2 after scaling */
	  zac2 = zabs2(c2);
	       
	  /* compute new A0 */
	  z3 = c2 * a2t * snlamt ;
	  t0[it0] = a0t * cslamt - IREAL * z3;
	       
	  /* compute new A2  */
	  r = zac2 * cslamt; 
	  z3 = CONJ(c2) * a0t * snlamt;
	  t2[it0] = ( a2t * r - IREAL * z3 ) * r_fratio;
        } /* end x for-loop */
      } /* end y for-loop */
    } /* end of distribute z loop */
  } /* end of OMP target */
  stop_omp_time();
}
