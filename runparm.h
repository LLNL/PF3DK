/* runparm.h */

#include "mytypes.h"

/* Storage for these variables needs to be allocated ONCE
   on the host. In the file responsible for allocating
   storage, define __parm_init__. */
#ifdef __parm_init__
#undef EXTERN
#define EXTERN
#else
#undef EXTERN
#define EXTERN extern
#endif


#ifndef _RUNPARM_H_

#ifdef OMP45
#pragma omp declare target
#endif
/******************************************************************************/
/* Run Parameters */
EXTERN int  iloop;                       /* number of time steps run per loop cycle */
EXTERN int  use_random;              /* random initial values if non-zero */
EXTERN int  nprint;                     /* when to print step information */
EXTERN int  step;                         /* number of time steps run */

EXTERN int  nx;                      /* global box bounds x-direction */
EXTERN int  ny;                      /* global box bounds y-direction */
EXTERN int  nz;                      /* global box bounds z-direction */
EXTERN int  nxg0;                    /* global box bounds on local processor x-dir*/
EXTERN int  nxg1;                    /* global box bounds on local processor x-dir*/
EXTERN int  nyg0;                    /* global box bounds on local processor y-dir*/
EXTERN int  nyg1;                    /* global box bounds on local processor y-dir*/
EXTERN int  nzg0;                    /* global box bounds on local processor z-dir*/
EXTERN int  nzg1;                    /* global box bounds on local processor z-dir*/
EXTERN int  nxl;                     /* processor box bounds x-direction */
EXTERN int  nyl;                     /* processor box bounds y-direction */
EXTERN int  nzl;                     /* processor box bounds z-direction */
EXTERN int  num_thr;                 /* number of OpenMP threads to use */

EXTERN int mp_rank;                  /* MPI rank of this process */
EXTERN int mp_size;                  /* number of MPI ranks */
EXTERN int mp_p;                     /* number of domains in x-direction */
EXTERN int mp_q;                     /* number of domains in y-direction */
EXTERN int mp_r;                     /* number of domains in z-direction */
EXTERN int mp_myp;                   /* x domain number */
EXTERN int mp_myq;                   /* y domain number */
EXTERN int mp_myr;                   /* z domain number */

EXTERN int  isubcycle;               /* light subcycles per hydro cycle */
EXTERN real dt;                      /* light time step*/
EXTERN real dthyd;                   /* hydro time step*/
EXTERN real dx;                      /* grid resolution x-direction */
EXTERN real dy;                      /* grid resolution y-direction */
EXTERN real dz;                      /* grid resolution z-direction */
EXTERN real lx;                      /* plasma length x-direction */
EXTERN real ly;                      /* plasma length y-direction */
EXTERN real lz;                      /* plasma length z-direction */
EXTERN real small;                   /* guard against division by zero */
EXTERN real current_time;            /* current time */
EXTERN real dt0;                     /* initial time step*/

EXTERN real clight;                  /* speed of light [cm/s] */
EXTERN real csound;                  /* speed of sound [cm/s] */
EXTERN real timps;                   /* conversion factor, code time to psec */
EXTERN real timunit;                 /* k0*Cs for 1 micron wavelength [s-1] */

EXTERN int  fft_pkg;                 /* selects FFT - langdon, schwartztrauber, fftw */
EXTERN int  fft_msg;                 /* selects FFT message passing scheme, send/recv or alltoall */

#define MAX_NTHREAD 256
EXTERN double startWallTime[MAX_NTHREAD];
EXTERN double endWallTime[MAX_NTHREAD];
EXTERN double totalWallTime[MAX_NTHREAD];

EXTERN int nthr_kern;
EXTERN int mpi_cnt;

EXTERN int num_teams; /* num_teams should be roughly equal to the number
                         of SMs on the Nvidia board */
#ifdef OMP45
#pragma omp end declare target
#endif

#endif

#define _RUNPARM_H_
