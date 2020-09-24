#include "mytypes.h"

/* Storage for these variables needs to be allocated ONCE
   on the host. In the file responsible for allocating
   storage, define __parm_init__. */
#ifdef __parm_init__
#define EXTERN
#else
#define EXTERN extern
#endif


#ifndef _PF3DBENCHVARS_H_

#ifdef OMP45
#pragma omp declare target
#endif

EXTERN int     ngrd;             /* number of guard cells */
EXTERN int     nxa, nya, nza;    /* allocated box dimensions */
EXTERN int     omp_maxthreads;

EXTERN long    ngtot, nplng, nmaxpln, ntheta;

/* xy arrays */
EXTERN real     *theta;        /* rotation angles */

/* 3D arrays */
EXTERN rcomplex *tN_new;       /* temp storage for a new light wave */
EXTERN real     *thetb;        /* angles as a function of z */
EXTERN real     *thetb_sav;    /* angles as a function of z backup*/

EXTERN char *cbuf;

#ifdef OMP45
#pragma omp end declare target
#endif

#endif

#define _PF3DBENCHVARS_H_
