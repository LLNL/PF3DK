/*
 * g3MPI.c
 *
 * this file provides the basic communications routines for pF3d
 * using MPI.
 *
 */

#include "grid.h"
#include "pf3dbench.h"
#include "runparm.h"
#include <stdio.h>

#ifdef MP_TRACE
#include <stdio.h>
FILE *mptfp;
char mptname[20];
#endif

extern int mp_p, mp_q, mp_r, mp_myp, mp_myq, mp_myr;

int     master = 0;         /* master node rank */
grid3d  g3;

int     dfile=FALSE;        /* write output debug messages to files or stdout */

#define inx(i,j,k) in[i+j*nxl+k*nxl*nyl]
#define outx(i,j,k) out[i+j*nxfull+k*nxfull*nrows]
#define rcvbufx(i,j,k) rcvbuf[i+j*nxl+k*nxl*nrows]
#define sndbufx(i,j,k) sndbuf[i+j*nxl+k*nxl*nrows]

#define iny(i,j,k) in[i+j*nyl+k*nyl*nxl]
#define outy(i,j,k) out[i+j*nxfull+k*nyfull*ncols]
#define rcvbufy(i,j,k) rcvbuf[i+j*nyl+k*nyl*ncols]
#define sndbufy(i,j,k) sndbuf[i+j*nyl+k*nyl*ncols]

/*------------------------------------------------------------------------
 * build_grid
 *
 * define a 3-D Cartesian topology atop the hardware
 */

/* (sep 27, 2001)
 * Latest releases of IBM SP2 systems have an MPI bug caused by reordering
 * MPI_Cart_create generates an invalid map internally, so data is passed
 * to the wrong task ids!  the solution is not to reorder.
 */
#ifdef ALLOW_REORDER
#define REORDER TRUE
#else
#define REORDER FALSE
#endif

void build_grid(void)
{
    int       dims[3] = {0, 0, 0};              /* global size of 3-d grid */
    int       coord[3] = {0, 0, 0};             /* local grid position */
    int       periods[3] = {TRUE, TRUE, TRUE};  /* periodic in each dimension */
    int       temp[3];                          /* for allocating x/y/z comms */

#ifdef USE_SCR
    (void) SCR_Init();
    scr_max_filename = SCR_MAX_FILENAME;
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &g3.me);
    MPI_Comm_size(MPI_COMM_WORLD, &g3.nproc);

#ifdef MP_TRACE
    sprintf(mptname, "MPTRACE%d.log", g3.me);
    mptfp = fopen(mptname,"w");
#endif

#if defined(DEBUG) && (DEBUG>128)
    /* print this out for debugging purposes */
    (void) printf("bld[%d]: Process %d, size %d\n", g3.me, g3.me, g3.nproc);
#endif
    
    dims[MP_X] = mp_p;
    dims[MP_Y] = mp_q;
    dims[MP_Z] = mp_r;
    MPI_Dims_create(g3.nproc, 3, dims);
    g3.P = dims[MP_X];
    g3.Q = dims[MP_Y];
    g3.R = dims[MP_Z];
    
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, REORDER, &g3.gridcom);
    MPI_Cart_get(g3.gridcom, 3, dims, periods, coord);
    g3.myP = coord[MP_X];
    g3.myQ = coord[MP_Y];
    g3.myR = coord[MP_Z];
    
#if defined(DEBUG) && (DEBUG>128)
    if (g3.me == master)
        (void) printf("bld[0]: size of mesh is %d by %d by %d\n", g3.P, g3.Q, g3.R);
    (void) printf("bld[%d]: coord is (%d,%d,%d)\n", g3.me, g3.myP, g3.myQ, g3.myR);
#endif
    
    /* get six neighbors (0=W, 1=E, 2=S, 3=N, 4=D, 5=U) */
    /* guard against negative node numbers because IBM MPI doesn't handle that */
    coord[MP_X] = ( g3.myP - 1 + g3.P ) % g3.P;
    coord[MP_Y] = g3.myQ;
    coord[MP_Z] = g3.myR;
    MPI_Cart_rank(g3.gridcom, coord, &g3.nbr[0]);

    coord[MP_X] = ( g3.myP + 1 ) % g3.P;
    coord[MP_Y] = g3.myQ;
    coord[MP_Z] = g3.myR;
    MPI_Cart_rank(g3.gridcom, coord, &g3.nbr[1]);

    coord[MP_X] = g3.myP;
    coord[MP_Y] = ( g3.myQ - 1 + g3.Q ) % g3.Q;
    coord[MP_Z] = g3.myR;
    MPI_Cart_rank(g3.gridcom, coord, &g3.nbr[2]);

    coord[MP_X] = g3.myP;
    coord[MP_Y] = ( g3.myQ + 1 ) % g3.Q;
    coord[MP_Z] = g3.myR;
    MPI_Cart_rank(g3.gridcom, coord, &g3.nbr[3]);
    
    coord[MP_X] = g3.myP;
    coord[MP_Y] = g3.myQ;
    coord[MP_Z] = ( g3.myR - 1 + g3.R ) % g3.R;
    MPI_Cart_rank(g3.gridcom, coord, &g3.nbr[4]);
    
    coord[MP_X] = g3.myP;
    coord[MP_Y] = g3.myQ;
    coord[MP_Z] = ( g3.myR + 1 ) % g3.R;
    MPI_Cart_rank(g3.gridcom, coord, &g3.nbr[5]);
    
#if defined(DEBUG) && (DEBUG>128)
    (void) printf("bld[%d]: nbrs = %d/%d/%d/%d/%d/%d (EWNSUD)\n", g3.me,
                  g3.nbr[1], g3.nbr[0], g3.nbr[3], g3.nbr[2], g3.nbr[5], g3.nbr[4]);
#endif

    /* define z communicator */
    temp[MP_X] = FALSE; temp[MP_Y] = FALSE; temp[MP_Z] = TRUE;
    MPI_Cart_sub(g3.gridcom, temp, &g3.zcom);

    /* define y communicator */
    temp[MP_X] = FALSE; temp[MP_Y] = TRUE; temp[MP_Z] = FALSE;
    MPI_Cart_sub(g3.gridcom, temp, &g3.ycom);

    /* define x communicator */
    temp[MP_X] = TRUE; temp[MP_Y] = FALSE; temp[MP_Z] = FALSE;
    MPI_Cart_sub(g3.gridcom, temp, &g3.xcom);

    /* get z neighbors (DOWN and UP) */
    coord[0] = ( g3.myR - 1 + g3.R ) % g3.R;
    MPI_Cart_rank(g3.zcom, coord, &g3.znbr[BEFORE]);
    coord[0] = ( g3.myR + 1 ) % g3.R;
    MPI_Cart_rank(g3.zcom, coord, &g3.znbr[AFTER]);

    /* define xy-communicator for sub-global collective operations */
    temp[MP_X] = TRUE;temp[MP_Y] = TRUE; temp[MP_Z] = FALSE;
    MPI_Cart_sub(g3.gridcom, temp, &g3.xycom);

    /* get diagonal xy neighbor */
#ifdef ORIG_ORDER
    coord[0] = (g3.P -1 - g3.myP);
    coord[1] = (g3.Q -1 - g3.myQ);
#else
    coord[1] = (g3.P -1 - g3.myP);
    coord[0] = (g3.Q -1 - g3.myQ);
#endif
    MPI_Cart_rank(g3.xycom, coord, &g3.diag);

    /* make some info available to the main program */
    mp_p   = g3.P;   mp_q   = g3.Q;   mp_r   = g3.R;
    mp_myp = g3.myP; mp_myq = g3.myQ; mp_myr = g3.myR;

#ifdef USE_GANG
    mp_gang_init();
#endif
}


/*------------------------------------------------------------------------
 * simabort
 *
 * abort the simulation
 */

void simabort(int errcode)
{
    MPI_Abort(MPI_COMM_WORLD, errcode);
}


/*------------------------------------------------------------------------
 * simallgather
 *
 * computes the global catenation of an 'n' vector 'localv' of reals into 'globalv'
 */

void simallgather(real *localv, real *globalv, int n)
{
    int i;
    double tim0, tim1;

    MPI_Allgather(localv, n, MPI_SINGLE, globalv, n, MPI_SINGLE, g3.gridcom);
}


/*------------------------------------------------------------------------
 * simbcast
 *
 * broadcast data from one node to entire grid
 */
void simbcast(void *in, int byte_count, int root)
{
    double tim0, tim1;

    MPI_Bcast(in, byte_count, MPI_BYTE, root, g3.gridcom);
}


/*------------------------------------------------------------------------
 * simdone
 *
 * simulation is finished, exit the message passing.
 */

void simdone(void)
{
    MPI_Finalize();
}


/*------------------------------------------------------------------------
 * simgather
 *
 * gather scalars from each node into a result vector on the master node
 */

void simgather(double *scalar, double *vector)
{
    double tim0, tim1;

    MPI_Gather(scalar, 1, MPI_DOUBLE, vector, 1, MPI_DOUBLE, master, g3.gridcom);
}


/*------------------------------------------------------------------------
 * siminit
 *
 * initialize parallel simulation 
 */

void siminit(int *ac, char **av[])
{
    int mpi_res;
  
    /* initialize; get rank and size information */
    mpi_res= MPI_Init(ac,av);
    if(mpi_res != MPI_SUCCESS) {
      printf("MPI did not initialize properly\n");
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &g3.me);
    MPI_Comm_size(MPI_COMM_WORLD, &g3.nproc);
}


/*------------------------------------------------------------------------
 * simreduce
 *
 * reduce scalars from each node into a scalar result on the master node
 */

void simreduce(double in, double *out)
{
    double    localin = in;
    double tim0, tim1;

    MPI_Reduce(&localin, out, 1, MPI_DOUBLE, MPI_MAX, master, g3.gridcom);
}


/*------------------------------------------------------------------------
 * alltoallshift
 *
 * Distribute the data currently on this processor to all other processors
 * in its "row" (dir==0) or to all other processors in its "column" (dir!=0) 
 */

void alltoallshift(real *in, real *out, int nchunk, int chunksize, int dir)
{
    MPI_Status is;

    /* The caller "promises" that the data in "in" is in the proper order for
       an alltoall. Each process in the x or y communicator will receive
       "chunksize" elements of size MPI_SINGLE. The bytes destined for process zero
       in the communicator are first in "in", the elements for process one
       follow those for process zero, etc.
       When the alltoall returns, the bytes from all processes, including this
       process, will be in "out". The bytes in "out" will be in "alltoall order".
    */
    
#ifdef MP_TRACE
    fprintf(mptfp, "process %d: %d bytes shifted\n", g3.me, ((long)count)*sizeof(real));
#endif
    
    switch (dir) {

    case 0:
        MPI_Alltoall(in, chunksize, MPI_SINGLE, out, chunksize, MPI_SINGLE, g3.xcom);
	break;

    default:
        MPI_Alltoall(in, chunksize, MPI_SINGLE, out, chunksize, MPI_SINGLE, g3.ycom);
	break;
    }
}

void cmp_alltoallshift(rcomplex *in, rcomplex *out, int nchunk, int chunksize, int dir)
{
    MPI_Status is;

    /* The caller "promises" that the data in "in" is in the proper order for
       an alltoall. Each process in the x or y communicator will receive
       "2*chunksize" elements of size MPI_SINGLE. The bytes destined for process zero
       in the communicator are first in "in", the elements for process one
       follow those for process zero, etc.
       When the alltoall returns, the bytes from all processes, including this
       process, will be in "out". The bytes in "out" will be in "alltoall order".
    */
    
#ifdef MP_TRACE
    fprintf(mptfp, "process %d: %d bytes shifted\n", g3.me, ((long)count)*sizeof(real));
#endif
    
    switch (dir) {

    case 0:
      MPI_Alltoall((real *)in, 2*chunksize, MPI_SINGLE, (real *)out, 2*chunksize, MPI_SINGLE, g3.xcom);
	break;

    default:
        MPI_Alltoall((real *)in, 2*chunksize, MPI_SINGLE, (real *)out, 2*chunksize, MPI_SINGLE, g3.ycom);
	break;
    }
}

/* NOTES
   For idir==0, in is nxl-by-nyl-by-nzl, out is nxfull-by-nrows-by nzl,
   sndbuf is nxl-by-nyl/2-by-nzl, and rcvbuf is nxl-by-nyl/2-by-nzl.
   Else, in is nyl-by-nxl-by-nzl, out is nyfull-by-ncols-by nzl,
   sndbuf is nyl-by-nxl/2-by-nzl, and rcvbuf is nyl-by-nxl/2-by-nzl.
   The number of elements in all arrays is the same in both cases.
*/
void cmp_msgshift(rcomplex *in, rcomplex *out, int idir)
{
    MPI_Status is;
    int i, j, k, count, nreal;

    /* The input data is in row-major or column-major order 
       depending on the value of idir. This function only works
       with a 2x2xN decomposition.
       Each process keeps half of its "rows" and sends the other
       half to its neighbor if idir==0 or keeps half of its
       "columns" sends the other half to theits neighbor.
       NOTE - this function currently assumes that the array
       is transposed before a call with idir != 0 so that, in both
       cases, stride one chunks are kept.
       COMMENT - MPI_Alltoall works well for sending data
       for one xy-plane. It requires many calls to send data from
       many xy-planes. This function sends data from all xy-planes
       using a single Send/Recv pair.
    */

    /* send half the bytes in the rcomplex array for a 2-by-2 decomp */
    nreal= nxl*nyl*nzl*2;
    count= nreal/2;
    if(idir == 0) {
        /* sending data from a row-major array */
        if(mp_myp | 1) {
            /* odd numbered x-ranks send the earlier half of their rows to
               the neighboring even numbered process. */
          for(k= 0; k < nzl; k++) {
              for(j= 0; j < nyl/2; j++) {
                    for(i= 0; i < nxl; i++) {
                        sndbufx(i,j,k)= inx(i,j,k);
                        outx(i+nxl,j,k)= inx(i,j+nyl/2,k);
                    }
                }
            }
        } else {
            /* even numbered x-ranks send the later half of their rows to
            the neighboring odd numbered process. */
            for(k= 0; k < nzl; k++) {
                for(j= 0; j < nyl/2; j++) {
                    for(i= 0; i < nxl; i++) {
                        outx(i,j,k)= inx(i,j,k);
                        sndbufx(i,j,k)= inx(i,j+nyl/2,k);
                    }
                }
            }
        }
    } else {
        /* sending data from a column-major array */
        if(mp_myq | 1) {
            /* odd numbered y-ranks send the earlier half of their rows to
               the neighboring even numbered process. */
            for(k= 0; k < nzl; k++) {
                for(j= 0; j < nxl/2; j++) {
                    for(i= 0; i < nyl; i++) {
                        sndbufy(i,j,k)= iny(i,j,k);
                        outy(i+nyl,j,k)= iny(i,j+nxl/2,k);
                    }
                }
            }
        } else {
            /* even numbered y-ranks send the later half of their rows to
               the neighboring odd numbered process. */
            for(k= 0; k < nzl; k++) {
                for(j= 0; j < nxl/2; j++) {
                    for(i= 0; i < nyl; i++) {
                        outy(i,j,k)= iny(i,j,k);
                        sndbufy(i,j,k)= iny(i,j+nxl/2,k);
                    }
                }
            }
        }
    }

    /* Send the bytes to the neighboring process */
    cmp_sendrecv(sndbuf, rcvbuf, count, idir);

    if(idir == 0) {
        /* receiving data from a row-major array */
        if(mp_myp | 1) {
            /* odd numbered x-ranks send the earlier half of their rows to
               the neighboring even numbered process. */
            for(k= 0; k < nzl; k++) {
                for(j= 0; j < nyl/2; j++) {
                    for(i= 0; i < nxl; i++) {
                        outx(i,j,k)= rcvbufx(i,j,k);
                    }
                }
            }
        } else {
            /* even numbered x-ranks send the later half of their rows to
            the neighboring odd numbered process. */
            for(k= 0; k < nzl; k++) {
                for(j= 0; j < nyl/2; j++) {
                    for(i= 0; i < nxl; i++) {
                        outx(i+nxl,j,k)= rcvbufx(i,j,k);
                    }
                }
            }
        }
    } else {
        /* receiving data from a column-major array */
        if(mp_myq | 1) {
            /* odd numbered y-ranks send the earlier half of their rows to
               the neighboring even numbered process. */
            for(k= 0; k < nzl; k++) {
                for(j= 0; j < nxl/2; j++) {
                    for(i= 0; i < nyl; i++) {
                        outy(i,j,k)= rcvbufy(i,j,k);
                    }
                }
            }
        } else {
            /* even numbered y-ranks send the later half of their rows to
               the neighboring odd numbered process. */
          for(k= 0; k < nzl; k++) {
                for(j= 0; j < nxl/2; j++) {
                    for(i= 0; i < nyl; i++) {
                        outy(i+nyl,j,k)= rcvbufy(i,j,k);
                    }
                }
            }
        }
    }
}

void cmp_sendrecv(rcomplex *sndbuf, rcomplex *rcvbuf, int count, int idir)
{
  int mp_partner;
  MPI_Request ir, is;
  MPI_Status istat;
  
  if(idir == 0) {
    /* messages exchanged in the x-direction */
    mp_partner= mp_rank;
    if(mp_myp & 1) mp_partner= mp_rank-1;
    else mp_partner= mp_rank+1;
  } else {
    /* messages exchanged in the y-direction */
    mp_partner= mp_rank;
    if(mp_myq & 1) mp_partner= mp_rank-2;
    else mp_partner= mp_rank+2;
  }
  #ifdef DBG_sndrecv
  printf("rank %d has partner %d for direction %d\n", mp_rank, mp_partner, idir);
  printf("message count is %d, zones per process is %d\n", count, nxl*nyl*nzl/2);
  fflush(0);
  #endif

  #ifdef MPI_MSG
  #define MPI_MSG
  MPI_Isend(sndbuf, count, MPI_SINGLE, mp_partner, MPI_ANY_TAG, g3.xcom, &is);
  MPI_Recv(rcvbuf, count, MPI_SINGLE, mp_partner, MPI_ANY_TAG, g3.xcom, &istat);
  MPI_Wait(&is, &istat);
  #else
  {
    int i;;

    /* count is number of MPI_SINGLE to copy, but sndbuf and rcvbuf
       are rcomplex. */
    for(i= 0; i < count/2; i++) {
      rcvbuf[i]= sndbuf[i];
    }
  }
  #endif
}


/*------------------------------------------------------------------------
 * simsync
 *
 * synchronize entire grid
 */
void simsync(void)
{
    double tim0, tim1;

    MPI_Barrier(g3.gridcom);
}
