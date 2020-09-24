/* light.h */

#include "mytypes.h"

/******************************************************************************/
/* Access macros */

/* light */
#define t0(a,b,c) t0[CELTNDX(a,b,c)]
#define t0_sav(a,b,c) t0_sav[CELTNDX(a,b,c)]
#define t2(a,b,c) t2[CELTNDX(a,b,c)]

/* t0_big has NO guard zones and is nxfull-by-nyfull zones in xy */
#define t0_big(i,j,k) t0_big[(((k)*nyfull + (j))*nxfull + (i))]

/* tN_new is nxl-by-nyl-by-nzl (no guard zones) */
#define tN_new(a,b,c) tN_new[CELTNDX3(a,b,c)]

/******************************************************************************/
/* Light Functions */

void reset_tvar(rcomplex * restrict tvar, rcomplex * restrict tvar_sav);
void reset_tvar_omp45(rcomplex * restrict tvar);
void get_tvar_omp45(rcomplex * restrict tvar);

void reset_t0_big(rcomplex * restrict tbig, rcomplex * restrict tbig_sav);


/* perform rotation using Buneman's method */
void rotth_z_merge(rcomplex * restrict ct0wk, real * restrict thetb, int izlo, int izhi);
void rotth_z_merge3(rcomplex * restrict ct0wk, real * restrict thetb, int izlo, int izhi);
#ifdef _OPENMP
#pragma omp declare target
#endif
void rotth_omp45(rcomplex * restrict tvar, real * restrict thetb, int iz);
void rotth_omp45_pre3D(int nzl, rcomplex * restrict tvar, real * restrict thetb,
                       int izlo, int izhi);
#ifdef _OPENMP
#pragma omp end declare target
#endif
void rotth_mult_omp45(int nzl, rcomplex * restrict tvar, real * restrict thetb);
void rotth_mult_omp45_pre(int nzl, rcomplex * restrict tvar,
                          real * restrict thetb);
void rotth_premap(rcomplex * restrict tvar, real * restrict thetb);
void rotth_unmap(rcomplex * restrict tvar, real * restrict thetb);

void couple_z(rcomplex * restrict t0, rcomplex * restrict t2,
               rcomplex * restrict denp);
void couple_z_merge3(rcomplex * restrict t0, rcomplex * restrict t2,
               rcomplex * restrict denp);
void couple_omp45(rcomplex * restrict t0, rcomplex * restrict t2, 
                  rcomplex * restrict denp);
void couple_premap(rcomplex * restrict t0, rcomplex * restrict t2,
                    rcomplex * restrict denp);
void couple_unmap(rcomplex * restrict t0, rcomplex * restrict t2,
                   rcomplex * restrict denp);
void couple_omp45_pre(rcomplex * restrict t0, rcomplex * restrict t2,
                       rcomplex * restrict denp);
void couple_omp45_pre_simd(rcomplex * restrict t0, rcomplex * restrict t2,
                            rcomplex * restrict denp);
