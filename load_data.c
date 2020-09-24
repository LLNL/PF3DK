#include "mytypes.h"
#include "util.h"
#include "load_data.h"

#define __parm_init__

#include "runparm.h"
#include "pf3dbench.h"

#include "pf3dbenchvars.h"


void load_scalars(void)
{
  /* Storage for "mirror variables" on the GPU was allocated
     before the program started. The variables must, of course,
     have been global or file scope so that they could be
     allocated at that point. The variables were given default
     values set on runparm.h etc. The problem is that many of
     those defaults are overridden by the pF3D input deck.

     This function is callled after the pF3D input deck has been
     read (or after init.c has been called in the case of
     pf3d-kern). It initializes scalars.
  */
#ifdef OMP45_BUILD
  load_scalars_omp45(nx, ny, nxl, nyl, nzl, nxa, nya,
                     nza, ngrd, nplng, ngtot, num_teams,
                     ntheta, clight, csound, dt, dx, dy, dz, timunit,
                     mp_rank, mp_size, mp_p, mp_q, mp_r, mp_myp, mp_myq, mp_myr);
#endif
}

void load_arrays(void)
{
  /* Storage for "mirror variables" on the GPU was allocated
     before the program started. The variables must, of course,
     have been global or file scope so that they could be
     allocated at that point. The variables were given default
     values set on runparm.h etc. The problem is that many of
     those defaults are overridden by the pF3D input deck.

     This function is callled after the pF3D input deck has been
     read (or after init.c has been called in the case of
     pf3d-kern). It sets up pointers on the device so that
     they point to the same data values as the host pointer. 
  */
#ifdef OMP45_BUILD
  load_arrays_omp45();
#endif
}

void parm_init(void)
{
  iloop=10;                 /* number of time steps run per loop cycle */
  use_random= 1;            /* random initial values if non-zero */
  nprint=100;               /* when to print step information */
  step=0;                   /* number of time steps run */
  
  small=1.e-7;              /* guard against division by zero */

  clight=3.e+10;            /* speed of light [cm/s] */
  csound=3.09e+7;           /* speed of sound [cm/s] */
  timps=0.0;                /* conversion factor, code time to psec */
  timunit=1.94e+12;         /* k0*Cs for 1 micron wavelength [s-1] */

#ifdef OMP45_BUILD
  num_teams= NUM_TEAMS;
#else
  num_teams= 1;
#endif
  mpi_cnt= 1;

  /* cbuf is scratch, so allocate but do not initialize */
  cbuf= (char *) malloc(NCBUF*sizeof(char));
}
