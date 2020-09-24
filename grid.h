/*
 * grid.h
 *
 * $Id: grid.h,v 1.7 2001/08/22 17:44:45 bert Exp $
 */
#include <mpi.h>

#include "mytypes.h"

#ifndef FALSE
#define FALSE (0)
#endif

#ifndef TRUE
#define TRUE (1)
#endif

/************************************************************************
 * macros
 ************************************************************************/
#ifdef USE_DOUBLE
#define MPI_SINGLE MPI_DOUBLE
#else
#define MPI_SINGLE MPI_FLOAT
#endif

/************************************************************************
 * parallel structure for communications
 ************************************************************************/
typedef struct {
    int      nproc;              /* number of processors == P * Q * R */
    int      P, Q, R;            /* global size of 2-d grid */
    int      myP, myQ, myR;      /* local grid position */
    int      me;                 /* local rank number */
    int      nbr[6];             /* nbr nodes (0=W, 1=E, 2=S, 3=N, 4=D, 5=U) */
    int      znbr[2];            /* neighnors in z */
    int      ynbr[2];            /* neighbor in y */
    int      xnbr[2];            /* neighbor in x */
    int      diag;               /* diagonal in xycom */
    int     *ndx0;               /* global low index for each X subdomain */
    int     *ndy0;               /* global low index for each Y subdomain */
    int     *ndz0;               /* global low index for each Z subdomain */
    MPI_Comm gridcom;            /* communicator for grid */
    MPI_Comm xcom;               /* communicator for x-direction */
    MPI_Comm ycom;               /* communicator for y-direction */
    MPI_Comm zcom;               /* communicator for z-direction */
    MPI_Comm xycom;              /* communicator for xy-direction */
    MPI_Datatype MPI_STATE;      /* Data typemap for state vector */  
} grid3d;


/************************************************************************
 * manifest constants
 ************************************************************************/

/* cardinal directions */
#define WEST  (0)
#define EAST  (1)
#define SOUTH (2)
#define NORTH (3)
#define DOWN  (4)
#define UP    (5)

#define DIAG  (6)

/* neighbor indexes */
#define BEFORE (0)
#define AFTER  (1)

/* communicator ordering */
#ifdef ORIG_ORDER
#define MP_X 0
#define MP_Y 1
#define MP_Z 2
#else
#define MP_X 2
#define MP_Y 1
#define MP_Z 0
#endif

/* message tags */
#define REPORT_TAG 65
#define LIGHT_TAG 68
#define SBS_TAG 69
#define SHIFT_TAG 70
#define GATHER_TAG 71
#define PCP_TAG 74
#define INFO_TAG 76
#define FLD_TAG 77
#define PRD_TAG 78
#define FXP_TAG 82
#define FYP_TAG 83
#define WX_TAG 84
#define WY_TAG 85
#define FXS_TAG 86
#define FYS_TAG 87

/************************************************************************
 * global variables
 ************************************************************************/

/* parallel parameters */
extern int     master;           /* master node rank */
extern grid3d  g3;

extern int     dfile;            /* write output debug messages to files or stdout */


/************************************************************************
 * functions 
 ************************************************************************/

/* define a 3-D Cartesian topology atop the hardware */
void build_grid(void);

/* abort the simulation */
void simabort(int errcode);

/* computes the global catenation of an 'n' vector 'localv' of reals into 'globalv' */
void simallgather(real *localv, real *globalv, int n);

/* broadcast data from one node to entire grid */
void simbcast(void *in, int byte_count, int root);

/* simulation is finished, exit the message passing. */
void simdone(void);

/* gather scalars from each node into a result vector on the master node */
void simgather(double *scalar, double *vector);

/* initialize parallel simulation */
void siminit(int *ac, char **av[]);

/* reduce scalars from each node into a scalar result on the master node */
void simreduce(double in, double *out);

/* synchronize entire grid */
void simsync(void);

/* Distribute the data currently on this processor to all other processors
   in its "row" or "column" */
void alltoallshift(real *in, real *out, int nchunk, int chunksize, int dir);
void cmp_alltoallshift(rcomplex *in, rcomplex *out, int nchunk,
                       int chunksize, int dir);
void cmp_sendrecv(rcomplex *sndbuf, rcomplex *rcvbuf, int nbytes, int idir);
void cmp_msgshift(rcomplex *in, rcomplex *out, int idir);
