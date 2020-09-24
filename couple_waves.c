/*
 * couple_waves.c
 *
 * Couple 3 waves contained in C99 complex arrays.
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

void couple_z(rcomplex * restrict t0, rcomplex * restrict t2,
              rcomplex * restrict denp)
{
  real  c20, cslamt, snlamt, r_zlam, r, fratio;
  real  r_fratio, cratio, zac2;
  double  zlam, c2re, c2im;
  rcomplex a0t, a2t, c2, z3;
  int     ix, iy, iz;
  long    it0;
  
  cratio= 1.0e3;
  fratio = SQRT(0.9);
  r_fratio = ONE/fratio;
  c20 = 0.25 * cratio * r_fratio;  

  start_omp_time();

#ifdef _OPENMP
  /* #pragma omp parallel for simd aligned(t0,t2:64) simdlen(real_lane_count) COLLAPSE(3) private(c2, a0t, a2t, zlam, r_zlam, snlamt, cslamt, r, z3, it0, zac2, c2re, c2im) */
#pragma omp parallel for COLLAPSE(2) private(c2, a0t, a2t, zlam, r_zlam, snlamt, cslamt, r, z3, it0, zac2, c2re, c2im)
#endif
  for (iz=0; iz<nzl; iz++) {
      for (iy=0; iy<nyl; iy++) {    
#ifdef _OPENMP
    #pragma omp simd aligned(t0,t2:64) simdlen(real_lane_count)
#endif
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
        } /* end for-loop */
      } /* end for-loop */
  }
  stop_omp_time();
}
