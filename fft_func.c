/* This file contains functions that perform 1D and 2D FFTs.
   There are versions that run in a single MPI process
   and versions that call MPI to pass messages when performing 2D FFTs.

   Most of these functions rely on helper functions in fft_util.c
*/

#ifdef _OPENMP
#include <omp.h>
#endif

#include <complex.h>
#include "fftw3.h"

#include "mytypes.h"
#include "pf3d_fft.h"
#include "grid.h"
#include "util.h"
#include <stdlib.h>

#include "runparm.h"
#include "pf3dbench.h"
#include "pf3dbenchvars.h"

fftw_plan plan_batch_1d_bkw_x, plan_batch_1d_fwd_x,
          plan_batch_1d_bkw_y, plan_batch_1d_fwd_y;
fftw_plan plan_batch_1d_bkw_x_mpi, plan_batch_1d_fwd_x_mpi,
          plan_batch_1d_bkw_y_mpi, plan_batch_1d_fwd_y_mpi;
fftw_plan plan_loop_1d_bkw_x, plan_loop_1d_fwd_x,
          plan_loop_1d_bkw_y, plan_loop_1d_fwd_y;
fftw_plan plan_loop_1d_bkw_x_loc, plan_loop_1d_fwd_x_loc,
          plan_loop_1d_bkw_y_loc, plan_loop_1d_fwd_y_loc;
fftw_plan plan_loop_1d_bkw_x_full, plan_loop_1d_fwd_x_full,
          plan_loop_1d_bkw_y_full, plan_loop_1d_fwd_y_full;
int ready_fft_batch= 0; /* tells whether fftw plans have been created */
int ready_fft_batch_mpi= 0; /* tells whether fftw plans have been created */
int ready_fft_loop= 0; /* tells whether fftw plans have been created */
int ready_fft_loop_loc= 0; /* tells whether fftw plans have been created */
int ready_fft_loop_full= 0; /* tells whether fftw plans have been created */

complex double *cmpvec_x= 0;
complex double *cmpvec_y= 0;

complex double *cmpvec_x_loc= 0;
complex double *cmpvec_y_loc= 0;

complex double *cmpvec_x_full= 0;
complex double *cmpvec_y_full= 0;

static double *tvec=0;
static fftw_plan plan_fwd= 0;
static fftw_plan plan_bkw= 0;

/* provide storage for multiple threads */
double *temp_vec= 0;

static fftw_plan plan_1D_x_fwd= 0;
static fftw_plan plan_1D_x_bkw= 0;
static fftw_plan plan_1D_y_fwd= 0;
static fftw_plan plan_1D_y_bkw= 0;

static int save_lenFFT_x=0;
static int save_lenFFT_y=0;

static int fft_tmplen= 0;

typedef struct {
  int nelem;
  fftw_plan plan_fwd;
  fftw_plan plan_bkw;
} fft_nplan;

extern double complex *tmp_dbcom;

#define grd2(i,j,k)  grd2[(((k)*(nyl+1) + (j))*(nxl+1) + (i))]
#define grd(i,j,k)  grd[(((k)*(nyl+1) + (j))*(nxl+1) + (i))]
#define dns(i,j,k)  dns[(((k)*nyl + (j))*nxl + (i))]
#define dns_v(i,j,k) dns[(((k)*num_j + (j))*num_i + (i))]
#define dbl(i,j,k)  dbl[(((k)*num_j + (j))*num_i + (i))]
/* The in() and out() macros are used when transposing arrays.
   Note the different order of the array dimensions. */
#define reg(i,j,k)  reg[(((k)*num_j + (j))*num_i + (i))]
#define trn(i,j,k)  trn[(((k)*num_i + (j))*num_j + (i))]
/* accessor macro */
#define var(a,b)  var[CELTNDX3(a,b,0)]
#define afftbuf(a,b,c)  afftbuf[a+b*nxl+c*nxl*nyl]
#define tmp_dbcom(a,b,c) tmp_dbcom[a+b*nxfull+c*nxfull*nyfull]

#undef DBG_ALLOC
#undef PRINT_VALS

/*
fftw_plan fftw_plan_many_dft(int rank, const int *n, int howmany,
                                  fftw_complex *in, const int *inembed,
                                  int istride, int idist,
                                  fftw_complex *out, const int *onembed,
                                  int ostride, int odist,
                                  int sign, unsigned flags);
*/

/*
  NOTES on arrys sizes etc.
  Light arrays like t0, t2, etc. have one guard zone in x and one
  guard zone in y.
  Temporary arrays used in fft_util do not have any guard zones.
  Both light arrays and FFT temporary arrays have nxl-by-nyl-by-nzl
  "real" zones.
  Light and temporary arrays are of type rcomplex. The FFT functions
  promote their arguments to double complex.

  The full grid is 2*nx-by-2*ny-by-nzl (real zones).
  nxfull=2*nx and nyfull=2*ny are the lengths of 1D FFTs.
  An MPI domain is nxl-by-nyl-by-nzl.

  A 2D FFT requires transposes so that MPI processes have entire
  rows or columns in their memory.
  When performing 1D FFTs in the x-direction, processes have
  nxfull-by-nrows-by-nzl arrays.
  When performing 1D FFTs in the y-direction, processes have
  nyfull-by-ncols-by-nzl arrays.
  nrows= nyl/mp_p
  ncols= nxl/mp_q

  nxfull= mp_p*nxl
  nyfull= mp_q*nyl

  so

  nxfull*nrows = nxl*mp_p*nyl/mp_p = nxl*nyl
  nyfull*nrcols = nyl*mp_q*nxl/mp_q = nxl*nyl

  This means that a domain has the same number of zones no matter
  what decomposition (rows, columns, checkerboard) is in use.
*/

#define USE_ALL2ALL
#define USE_FFT
#undef PRINT_VALS

void FFT_chek_x_loc(rcomplex *grd, int isign)
{
    int iz;

    start_omp_time();

    /* Perform a 1D FFT in the x-direction. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    /* copy data to a complex double temporary array */
    dense2dbl(afftbuf, tmp_dbcom, nxl, nyl, nzl);
    /* Perform nyl 1D FFTs of length nxl in nzl planes */
    for(iz= 0; iz < nzl; iz++) {
      fft_1D_loop_x_loc(tmp_dbcom, nxl, nyl, isign, iz);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nxl, nyl, nzl);
    /* Put the data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void FFT_chek_y_loc(rcomplex *grd, int isign)
{
    int iz;

    start_omp_time();

    /* Perform a 1D FFT in the y-direction. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    /* copy data to a complex double temporary array */
    dense2dbl(afftbuf, tmp_dbcom, nxl, nyl, nzl);
    /* Perform nyl 1D FFTs of length nxl in nzl planes */
    for(iz= 0; iz < nzl; iz++) {
      fft_1D_loop_y_loc(tmp_dbcom, nyl, nxl, isign, iz);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nxl, nyl, nzl);
    /* Put the data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void FFT_chek_loc(rcomplex *grd, int isign)
{
    rcomplex *obuf = (rcomplex *)optmp;
    int iz;

    start_omp_time();

    /* Perform a 2D FFT. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);

    /* copy data to a complex double temporary array */
    dense2dbl(afftbuf, tmp_dbcom, nxl, nyl, nzl);
    /* Perform nyl 1D FFTs of length nxl in nzl planes */
    for(iz= 0; iz < nzl; iz++) {
      fft_1D_loop_x_loc(tmp_dbcom, nxl, nyl, isign, iz);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nxl, nyl, nzl);

    /* Transpose to a "y varies fastest" checkerboard. */
    trans_to(afftbuf, obuf, nxl, nyl, nzl);
    /* copy data to a complex double temporary array */
    dense2dbl(obuf, tmp_dbcom, nyl, nxl, nzl);
    /* Perform nyl 1D FFTs of length nxl in nzl planes */
    for(iz= 0; iz < nzl; iz++) {
      fft_1D_loop_y_loc(tmp_dbcom, nyl, nxl, isign, iz);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nyl, nxl, nzl);
    /* Transpose to a "x varies fastest" checkerboard. */
    trans_from(afftbuf, obuf, nxl, nyl, nzl);

    /* Put the data back in the original array */
    dense2guard(obuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void FFT_2d_many(rcomplex *grd, int isign)
{
    rcomplex *obuf = (rcomplex *)optmp;
    int iz;

    start_omp_time();

    /* Perform a 2D FFT. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);

    /* copy data to a complex double temporary array */
    dense2dbl(afftbuf, tmp_dbcom, nxl, nyl, nzl);
    /* Perform nyl*nzl 1D FFTs of length nxl */
    if(isign == 1) {
	/* Perform a set of backward 1D FFTs */
        fftw_execute(plan_batch_1d_bkw_x);
    } else {
	/* Perform a set of forward 1D FFTs */
        fftw_execute(plan_batch_1d_fwd_x);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nxl, nyl, nzl);

    /* Transpose to a "y varies fastest" checkerboard. */
    trans_to(afftbuf, obuf, nxl, nyl, nzl);
    /* copy data to a complex double temporary array */
    dense2dbl(obuf, tmp_dbcom, nyl, nxl, nzl);
    /* Perform nxl*nzl 1D FFTs of length nyl */
    if(isign == 1) {
	/* Perform a set of backward 1D FFTs */
        fftw_execute(plan_batch_1d_bkw_y);
    } else {
	/* Perform a set of forward 1D FFTs */
        fftw_execute(plan_batch_1d_fwd_y);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nyl, nxl, nzl);
    /* Transpose to a "x varies fastest" checkerboard. */
    trans_from(afftbuf, obuf, nxl, nyl, nzl);

    /* Put the data back in the original array */
    dense2guard(obuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void FFT_chek_x(rcomplex *grd, int isign)
{
    rcomplex *obuf = (rcomplex *)optmp;
    int iz;
    long izoff;

    start_omp_time();

    /* Perform a 2D FFT. Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
#ifndef USE_MPI
    puts("WARNING - alltoall_chek only works in an MPI code");
#else
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(afftbuf+izoff, obuf+izoff, mp_p, chunksize_x, 0);
    }
    rows_to(obuf, afftbuf, chunksize_x, nrows, mp_p);
    /* copy data to a complex double temporary array */
    dense2dbl(afftbuf, tmp_dbcom, nxfull, nrows, nzl);
    /* Perform nrows 1D FFTs of length nxfull */
    for(iz= 0; iz < nzl; iz++) {
      fft_1D_loop_x_full(tmp_dbcom, nxfull, nrows, isign, iz);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nxfull, nrows, nzl);
    rows_from(afftbuf, obuf, chunksize_x, nrows, mp_p);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_p, chunksize_x, 0);
    }
#endif
    /* Put the data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void FFT_chek_y(rcomplex *grd, int isign)
{
    rcomplex *obuf = (rcomplex *)optmp;
    int iz;
    long izoff;

    start_omp_time();

    /* Perform a 2D FFT. Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    /* Transpose to a "y varies fastest" checkerboard. */
    trans_to(afftbuf, obuf, nxl, nyl, nzl);
#ifndef USE_MPI
    puts("WARNING - alltoall_chek only works in an MPI code");
    return;
#else
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_q, chunksize_y, 1);
    }
    cols_to(afftbuf, obuf, chunksize_y, ncols, mp_q);
    /* copy data to a complex double temporary array */
    dense2dbl(obuf, tmp_dbcom, nyfull, ncols, nzl);
    /* Perform nrcol 1D FFTs of length nyfull */
    for(iz= 0; iz < nzl; iz++) {
      fft_1D_loop_y_full(tmp_dbcom, nyfull, ncols, isign, iz);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nyfull, ncols, nzl);
    cols_from(afftbuf, obuf, chunksize_y, ncols, mp_q);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_q, chunksize_y, 1);
    }
#endif
    /* Transpose to a "x varies fastest" checkerboard. */
    trans_from(afftbuf, obuf, nxl, nyl, nzl);
    /* Put the data back in the original array */
    dense2guard(obuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void FFT1D_x(rcomplex *grd, int isign)
{
    rcomplex *obuf = (rcomplex *)optmp;
    int iz;
    long izoff;

    start_omp_time();

#ifndef USE_MPI
    puts("WARNING - alltoall_chek only works in an MPI code");
    return;
#endif
    
    /* Perform a 2D FFT. Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(afftbuf+izoff, obuf+izoff, mp_p, chunksize_x, 0);
    }
    rows_to(obuf, afftbuf, chunksize_x, nrows, mp_p);
    /* copy data to a complex double temporary array */
    dense2dbl(afftbuf, tmp_dbcom, nxfull, nrows, nzl);
    /* Perform nrows 1D FFTs of length nxfull */
    for(iz= 0; iz < nzl; iz++) {
      fft_1D_loop_x_full(tmp_dbcom, nxfull, nrows, isign, iz);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nxfull, nrows, nzl);
    rows_from(afftbuf, obuf, chunksize_x, nrows, mp_p);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_p, chunksize_x, 0);
    }
    /* Put the un-normalized FFT data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);

    stop_omp_time();
}

void FFT2D(rcomplex *grd, int isign)
{
    rcomplex *obuf = (rcomplex *)optmp;
    int iz;
    long izoff;

    start_omp_time();

#ifndef USE_MPI
    puts("WARNING - alltoall_chek only works in an MPI code");
    return;
#endif
    
    /* Perform a 2D FFT. Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(afftbuf+izoff, obuf+izoff, mp_p, chunksize_x, 0);
    }
    rows_to(obuf, afftbuf, chunksize_x, nrows, mp_p);
    /* copy data to a complex double temporary array */
    dense2dbl(afftbuf, tmp_dbcom, nxfull, nrows, nzl);
    /* Perform nrows 1D FFTs of length nxfull */
    for(iz= 0; iz < nzl; iz++) {
      fft_1D_loop_x_full(tmp_dbcom, nxfull, nrows, isign, iz);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nxfull, nrows, nzl);
    rows_from(afftbuf, obuf, chunksize_x, nrows, mp_p);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_p, chunksize_x, 0);
    }
    /* Put the un-normalized FFT data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);

    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    /* Transpose to a "y varies fastest" checkerboard. */
    trans_to(afftbuf, obuf, nxl, nyl, nzl);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_q, chunksize_y, 1);
    }
    cols_to(afftbuf, obuf, chunksize_y, ncols, mp_q);
    /* copy data to a complex double temporary array */
    dense2dbl(obuf, tmp_dbcom, nyfull, ncols, nzl);
    /* Perform nrcol 1D FFTs of length nyfull */
    for(iz= 0; iz < nzl; iz++) {
      fft_1D_loop_y_full(tmp_dbcom, nyfull, ncols, isign, iz);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nyfull, ncols, nzl);
    cols_from(afftbuf, obuf, chunksize_y, ncols, mp_q);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_q, chunksize_y, 1);
    }
    /* Transpose to a "x varies fastest" checkerboard. */
    trans_from(afftbuf, obuf, nxl, nyl, nzl);
    /* Put the un-normalized FFT data back in the original array */
    dense2guard(obuf, grd, nxl, nyl, nzl);

    stop_omp_time();
}

void FFT2D_noguard(rcomplex *tvar, int isign)
{
    rcomplex *obuf = (rcomplex *)optmp;
    int iz;
    long izoff;

    start_omp_time();

#ifndef USE_MPI
    puts("WARNING - alltoall_chek only works in an MPI code");
    return;
#endif
    
    /* Perform a 2D FFT. Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(tvar+izoff, obuf+izoff, mp_p, chunksize_x, 0);
    }
    rows_to(obuf, afftbuf, chunksize_x, nrows, mp_p);
    /* copy data to a complex double temporary array */
    dense2dbl(afftbuf, tmp_dbcom, nxfull, nrows, nzl);
    /* Perform nrows 1D FFTs of length nxfull */
    for(iz= 0; iz < nzl; iz++) {
      fft_1D_loop_x_full(tmp_dbcom, nxfull, nrows, isign, iz);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nxfull, nrows, nzl);
    rows_from(afftbuf, obuf, chunksize_x, nrows, mp_p);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_p, chunksize_x, 0);
    }

    /* Transpose to a "y varies fastest" checkerboard. */
    trans_to(afftbuf, obuf, nxl, nyl, nzl);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_q, chunksize_y, 1);
    }
    cols_to(afftbuf, obuf, chunksize_y, ncols, mp_q);
    /* copy data to a complex double temporary array */
    dense2dbl(obuf, tmp_dbcom, nyfull, ncols, nzl);
    /* Perform nrcol 1D FFTs of length nyfull */
    for(iz= 0; iz < nzl; iz++) {
      fft_1D_loop_y_full(tmp_dbcom, nyfull, ncols, isign, iz);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nyfull, ncols, nzl);
    cols_from(afftbuf, obuf, chunksize_y, ncols, mp_q);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_q, chunksize_y, 1);
    }
    /* Transpose to a "x varies fastest" checkerboard. */
    trans_from(afftbuf, tvar, nxl, nyl, nzl);

    stop_omp_time();
}


/* Complex double 1D Fourier transform */
int fft_1D_loop_x_full(complex double *var, int lenFFT, int numFFT, int isign,
                      int iz)
{
    long i, j, izoff;

    start_omp_time();

    /* WARNING - this function may only be called with the array
       used in the prepFFT_loop_full call. */
    /*
     *  var	data (complex var(lenFFT,numFFT) in calling program)
     *  isign	isign in complex exponential in Fourier transform
     *
     *  The FFT is performed in place.
     */
    /* NOTE: The convention for isign in this routine is 
       different from that in the yorick FFT package and
       that in FFTW.
       It is set to be compatible with Langdon's FFT.
     */

    /* create an FFTW plan for a 1D FFT, if needed */
    if(!ready_fft_loop_full) {
        /* This assumes that one xy-plane of FFTs will be run at one time */ 
      prepFFT_loop_full(nxfull, nrows, nyfull, ncols);
    }

#ifdef DEBUG_FFT
    if(lenFFT != nxfull) printf("fft_1D_loop_x_full: lenFFT=%d, nxfull=%d\n", lenFFT, nxfull);
    if(!mp_rank && iz == 0) {
      printf("fft_1D_loop_x_full: lenFFT=%d, numFFT=%d, iz=%d, nxfull=%d\n", lenFFT, numFFT, iz, nxfull);
    }
#endif
    izoff= iz*lenFFT*numFFT;
    /* Perform a set of backward 1D FFTs */
    for(j= 0; j < numFFT; j++) {
      for(i= 0; i < lenFFT; i++) {
        cmpvec_x_full[i]= var[i+j*lenFFT+izoff];
      }
      if(isign == 1) {
        fftw_execute(plan_loop_1d_bkw_x_full);
      } else {
        fftw_execute(plan_loop_1d_fwd_x_full);
      }
      for(i= 0; i < lenFFT; i++) {
        var[i+j*lenFFT+izoff]= cmpvec_x_full[i];
      }
    }
    stop_omp_time();
    return(0);  /* return 0 to indicate success */
}

/* Complex double 1D Fourier transform */
int fft_1D_loop_y_full(complex double *var, int lenFFT, int numFFT, int isign,
                      int iz)
{
    long i, j, izoff;

    start_omp_time();

    /* WARNING - this function may only be called with the array
       used in the prepFFT_loop_full call. */
    /*
     *  var	data (complex var(lenFFT,numFFT) in calling program)
     *  isign	isign in complex exponential in Fourier transform
     *
     *  The FFT is performed in place.
     */
    /* NOTE: The convention for isign in this routine is 
       different from that in the yorick FFT package and
       that in FFTW.
       It is set to be compatible with Langdon's FFT.
     */

   /* create an FFTW plan for a 1D FFT, if needed */
    if(!ready_fft_loop_full) {
        /* This assumes that one xy-plane of FFTs will be run at one time */ 
      prepFFT_loop_full(nxfull, nrows, nyfull, ncols);
    }

#ifdef DEBUG_FFT
    if(lenFFT != nyfull) printf("fft_1D_loop_x_full: lenFFT=%d, nyfull=%d\n", lenFFT, nyfull);
#endif
    izoff= iz*lenFFT*numFFT;
    /* Perform a set of backward 1D FFTs */
    for(j= 0; j < numFFT; j++) {
      for(i= 0; i < lenFFT; i++) {
        cmpvec_y_full[i]= var[i+j*lenFFT+izoff];
      }
      if(isign == 1) {
        fftw_execute(plan_loop_1d_bkw_y_full);
      } else {
        fftw_execute(plan_loop_1d_fwd_y_full);
      }
      for(i= 0; i < lenFFT; i++) {
        var[i+j*lenFFT+izoff]= cmpvec_y_full[i];
      }
    }
    stop_omp_time();
    return(0);  /* return 0 to indicate success */
}

/* Complex double 1D Fourier transform */
int fft_1D_loop_x_loc(complex double *var, int lenFFT, int numFFT, int isign,
                      int iz)
{
    long i, j, izoff;

    start_omp_time();

    /* WARNING - this function may only be called with the array
       used in the prepFFT_loop_loc call. */
    /*
     *  var	data (complex var(lenFFT,numFFT) in calling program)
     *  isign	isign in complex exponential in Fourier transform
     *
     *  The FFT is performed in place.
     */
    /* NOTE: The convention for isign in this routine is 
       different from that in the yorick FFT package and
       that in FFTW.
       It is set to be compatible with Langdon's FFT.
     */

    /* create an FFTW plan for a 1D FFT, if needed */
    if(!ready_fft_loop_loc) {
        /* This assumes that one xy-plane of FFTs will be run at one time */ 
        prepFFT_loop_loc(nxl, nyl, nyl, nxl);
    }

    izoff= iz*lenFFT*numFFT;
    if(isign == 1) {
	/* Perform a set of backward 1D FFTs */
      for(j= 0; j < numFFT; j++) {
        for(i= 0; i < lenFFT; i++) {
          cmpvec_x_loc[i]= var[i+j*lenFFT+izoff];
        }
        fftw_execute(plan_loop_1d_bkw_x_loc);
        for(i= 0; i < lenFFT; i++) {
          var[i+j*lenFFT+izoff]= cmpvec_x_loc[i];
        }
      }
    } else {
      for(j= 0; j < numFFT; j++) {
	/* Perform a set of forward 1D FFTs */
        for(i= 0; i < lenFFT; i++) {
          cmpvec_x_loc[i]= var[i+j*lenFFT+izoff];
        }
        fftw_execute(plan_loop_1d_fwd_x_loc);
        for(i= 0; i < lenFFT; i++) {
          var[i+j*lenFFT+izoff]= cmpvec_x_loc[i];
        }
      }
    }
    stop_omp_time();
    return(0);  /* return 0 to indicate success */
}

/* Complex double 1D Fourier transform */
int fft_1D_loop_y_loc(complex double *var, int lenFFT, int numFFT, int isign,
                      int iz)
{
    long i, j, izoff;

    start_omp_time();

    /* WARNING - this function may only be called with the array
       used in the prepFFT_loop_loc call. */
    /*
     *  var	data (complex var(lenFFT,numFFT) in calling program)
     *  isign	isign in complex exponential in Fourier transform
     *
     *  The FFT is performed in place.
     */
    /* NOTE: The convention for isign in this routine is 
       different from that in the yorick FFT package and
       that in FFTW.
       It is set to be compatible with Langdon's FFT.
     */

    /* create an FFTW plan for a 1D FFT, if needed */
    if(!ready_fft_loop_loc) {
        /* This assumes that one xy-plane of FFTs will be run at one time */ 
        prepFFT_loop_loc(nxl, nyl, nyl, nxl);
    }

    izoff= iz*lenFFT*numFFT;
    if(isign == 1) {
	/* Perform a set of backward 1D FFTs */
      for(j= 0; j < numFFT; j++) {
        for(i= 0; i < lenFFT; i++) {
          cmpvec_y_loc[i]= var[i+j*lenFFT+izoff];
        }
        fftw_execute(plan_loop_1d_bkw_y_loc);
        for(i= 0; i < lenFFT; i++) {
          var[i+j*lenFFT+izoff]= cmpvec_y_loc[i];
        }
      }
    } else {
      for(j= 0; j < numFFT; j++) {
	/* Perform a set of forward 1D FFTs */
        for(i= 0; i < lenFFT; i++) {
          cmpvec_y_loc[i]= var[i+j*lenFFT+izoff];
        }
        fftw_execute(plan_loop_1d_fwd_y_loc);
        for(i= 0; i < lenFFT; i++) {
          var[i+j*lenFFT+izoff]= cmpvec_y_loc[i];
        }
      }
    }
    stop_omp_time();
    return(0);  /* return 0 to indicate success */
}

/* Complex double 1D Fourier transform */
int fft_1D_loop_x(complex double *var, int lenFFT, int numFFT,
                  int isign, int iz)
{
    long i, j, izoff;

    start_omp_time();

    /* WARNING - this function may only be called with the array
       used in the prepFFT_loop call. */
    /*
     *  var	data (complex var(lenFFT,numFFT) in calling program)
     *  isign	isign in complex exponential in Fourier transform
     *
     *  The FFT is performed in place.
     */
    /* NOTE: The convention for isign in this routine is 
       different from that in the yorick FFT package and
       that in FFTW.
       It is set to be compatible with Langdon's FFT.
     */

   /* create an FFTW plan for 1D FFTs, if needed */
    if(!ready_fft_loop) {
        /* This assumes that one xy-plane of FFTs will be run at a time */ 
        prepFFT_loop(nxfull, nrows, nyfull, ncols);
        printf("fftw plans have been prepared for fft_1D_loop_x\n");
        printf("var=%p, cmpvec_x=%p\n", var, cmpvec_x);
    }

#ifdef DEBUG_FFT
    if(lenFFT != nxfull) printf("fft_1D_loop_x: lenFFT=%d, nxfull=%d\n", lenFFT, nxfull);
#endif
    izoff= iz*lenFFT*numFFT;
    if(isign == 1) {
	/* Perform a set of backward 1D FFTs */
      for(j= 0; j < numFFT; j++) {
        for(i= 0; i < lenFFT; i++) {
          cmpvec_x[i]= var[i+j*lenFFT+izoff];
        }
        fftw_execute(plan_loop_1d_bkw_x);
        for(i= 0; i < lenFFT; i++) {
          var[i+j*lenFFT+izoff]= cmpvec_x[i];
        }
      }
    } else {
      for(j= 0; j < numFFT; j++) {
	/* Perform a set of forward 1D FFTs */
        for(i= 0; i < lenFFT; i++) {
          cmpvec_x[i]= var[i+j*lenFFT+izoff];
        }
        fftw_execute(plan_loop_1d_fwd_x);
        for(i= 0; i < lenFFT; i++) {
          var[i+j*lenFFT+izoff]= cmpvec_x[i];
        }
      }
    }
    stop_omp_time();
    return(0);  /* return 0 to indicate success */
}

/* Complex double 1D Fourier transform */
int fft_1D_loop_y(complex double *var, int lenFFT, int numFFT,
                  int isign, int iz)
{
    long i, j, izoff;

    start_omp_time();

    /* WARNING - this function may only be called with the array
       used in the prepFFT_loop call. */
    /*
     *  var	data (complex var(lenFFT,numFFT) in calling program)
     *  isign	isign in complex exponential in Fourier transform
     *
     *  The FFT is performed in place.
     */
    /* NOTE: The convention for isign in this routine is 
       different from that in the yorick FFT package and
       that in FFTW.
       It is set to be compatible with Langdon's FFT.
     */

    /* create an FFTW plan for a batch of 1D FFTs, if needed */
    if(!ready_fft_loop) {
        /* This assumes that one xy-plane of FFTs will be run at one time */ 
        prepFFT_loop(nxfull, nrows, nyfull, ncols);
    }

#ifdef DEBUG_FFT
    if(lenFFT != nyfull) printf("fft_1D_loop_y: lenFFT=%d, nyfull=%d\n", lenFFT, nyfull);
#endif
    izoff= iz*lenFFT*numFFT;
    if(isign == 1) {
	/* Perform a set of backward 1D FFTs */
      for(j= 0; j < numFFT; j++) {
        for(i= 0; i < lenFFT; i++) {
          cmpvec_y[i]= var[i+j*lenFFT+izoff];
        }
        fftw_execute(plan_loop_1d_bkw_y);
        for(i= 0; i < lenFFT; i++) {
          var[i+j*lenFFT+izoff]= cmpvec_x[i];
        }
      }
    } else {
	/* Perform a set of forward 1D FFTs */
      for(j= 0; j < numFFT; j++) {
        for(i= 0; i < lenFFT; i++) {
          cmpvec_y[i]= var[i+j*lenFFT+izoff];
        }
        fftw_execute(plan_loop_1d_fwd_y);
        for(i= 0; i < lenFFT; i++) {
          var[i+j*lenFFT+izoff]= cmpvec_x[i];
        }
      }
    }
    stop_omp_time();
    return(0);  /* return 0 to indicate success */
}

/* Complex double 1D Fourier transform */
int fft_1D_batch_x(rcomplex *var, int lenFFT, int numFFT,
                   int isign, int nzl)
{
    long i, j, izoff;

    start_omp_time();

    /* WARNING - this function may only be called with the array
       used in the prepBatchFFT call. */
    /*
     *  var	data (complex var(lenFFT,numFFT) in calling program)
     *  isign	isign in complex exponential in Fourier transform
     *
     *  The FFT is performed in place.
     */
    /* NOTE: The convention for isign in this routine is 
       different from that in the yorick FFT package and
       that in FFTW.
       It is set to be compatible with Langdon's FFT.
     */

    /* create an FFTW plan for a batch of 1D FFTs, if needed */
    if(!ready_fft_batch) {
        /* the plan must use the double complex temporary array, not
           the input array, because the input often has float precision. */
        prepBatchFFT(nxl, nyl*nzl, nyl, nxl*nzl, tmp_dbcom);
    }
#ifdef DEBUG_FFT
    if(lenFFT != nxl) printf("fft_1D_batch_x: lenFFT=%d, nxl=%d\n", lenFFT, nxl);
#endif
    
    /* copy without guard zones */
    guard2dense(var, afftbuf, nxl, nyl, nzl);
    /* copy the values from the input to a densely packed double complex
       array */
    dense2dbl(afftbuf, tmp_dbcom, nxl, nyl, nzl);
    if(isign == 1) {
	/* Perform a set of backward 1D FFTs */
        fftw_execute(plan_batch_1d_bkw_x);
    } else {
	/* Perform a set of forward 1D FFTs */
        fftw_execute(plan_batch_1d_fwd_x);
    }
    /* copy the results back to the input array. */
    dbl2dense(tmp_dbcom, afftbuf, nxl, nyl, nzl);
    dense2guard(afftbuf, var, nxl, nyl, nzl);
    stop_omp_time();
    return(0);  /* return 0 to indicate success */
}

/* Complex double 1D Fourier transform */
int fft_1D_batch_y(rcomplex *var, int lenFFT, int numFFT,
                   int isign, int nzl)
{
    long i, j, izoff;
    rcomplex *obuf = (rcomplex *)optmp;

    start_omp_time();

    /* WARNING - this function may only be called with the array
       used in the prepBatchFFT call. */
    /*
     *  var	data (complex var(lenFFT,numFFT) in calling program)
     *  isign	isign in complex exponential in Fourier transform
     *
     *  The FFT is performed in place.
     */
    /* NOTE: The convention for isign in this routine is 
       different from that in the yorick FFT package and
       that in FFTW.
       It is set to be compatible with Langdon's FFT.
     */

    /* create an FFTW plan for a batch of 1D FFTs, if needed */
    if(!ready_fft_batch) {
        prepBatchFFT(nxl, nyl*nzl, nyl, nxl*nzl, tmp_dbcom);
    }
#ifdef DEBUG_FFT
    if(lenFFT != nyl) printf("fft_1D_batch_y: lenFFT=%d, nyl=%d\n", lenFFT, nyl);
#endif
    
    /* remove the guard zones from the input array */
    guard2dense(var, afftbuf, nxl, nyl, nzl);
    trans_to(afftbuf, obuf, nxl, nyl, nzl);
    /* copy data to a complex double temporary array */
    dense2dbl(obuf, tmp_dbcom, nyl, nxl, nzl);
    if(isign == 1) {
	/* Perform a set of backward 1D FFTs */
        fftw_execute(plan_batch_1d_bkw_y);
    } else {
	/* Perform a set of forward 1D FFTs */
        fftw_execute(plan_batch_1d_fwd_y);
    }
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, obuf, nyl, nxl, nzl);
    trans_from(obuf, afftbuf, nxl, nyl, nzl);
    /* Put the data back in the original array */
    dense2guard(afftbuf, var, nxl, nyl, nzl);
    stop_omp_time();
    return(0);  /* return 0 to indicate success */
}

/* Complex double 1D Fourier transform */
int fft_1D_batch_mpi_x(rcomplex *var, int lenFFT, int numFFT,
                       int isign, int nxl, int nyl, int nzl)
{
    /*
     *  isign	sign in complex exponential in Fourier transform
     *  nxd	first dimension range as declared by caller (should be the same as nx)
     */
    int        incp, ier;
    rcomplex *obuf = (rcomplex *)optmp;
    rcomplex *ibuf = (rcomplex *)iptmp;
    rcomplex  *tbuf;
    int        bufoff, rowoff, coloff;
    int        ip, iq, i, j, iz, izoff;

    start_omp_time();

    /* WARNING - this function may only be called with the array
       used in the prepBatchFFT_MPI call. */
    /*
     *  var	data (complex var(lenFFT,numFFT) in calling program)
     *  isign	isign in complex exponential in Fourier transform
     *
     *  The FFT is performed in place.
     */
    /* NOTE: The convention for isign in this routine is 
       different from that in the yorick FFT package and
       that in FFTW.
       It is set to be compatible with Langdon's FFT.
     */

    /* create an FFTW plan for a batch of 1D FFTs, if needed */
    if(!ready_fft_batch_mpi) {
        /* the plan must use the double complex temporary array, not
           the input array, because the input often has float precision. */
        prepBatchFFT_MPI(lenFFT, numFFT*nzl, nyfull, nyl/mp_p*nzl, tmp_dbcom);
    }
#ifdef DEBUG_FFT
    if(lenFFT != nxfull) printf("fft_1D_batch_mpi_x: lenFFT=%d, nxfull=%d\n", lenFFT, nxfull);
#endif
  
    /***********************************************************************
     ****                  X Direction sweep                            ****
     ***********************************************************************/

    /*.................... Cycle through all P rows to collect columns 
     * The basic idea is to fill afftbufx with full rows of length lenFFT elements.
     * The number of rows per processor is nyl/mp_p */

    /* An MPI_Alltoall will be used to move the data into obuf.
        The data in "e" is already in the proper order, so use "e"
       as the "input buffer" for the message passing. */
    guard2dense(var, afftbuf , nxl, nyl, nzl);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(afftbuf+izoff, obuf+izoff, mp_p, chunksize_x, 0);
    }
    /*
     * obuf is nxl by nrows by mp_p
     * copy into afftbuf which is nxfull by nrows
     */
    rows_to(obuf, afftbuf, chunksize_x, nrows, mp_p);
                
    /*.................... Batched 1d FFT in X-direction */
    dense2dbl(afftbuf, tmp_dbcom, nxfull, nrows, nzl);
    if(isign == 1) {
	/* Perform a set of backward 1D FFTs */
        fftw_execute(plan_batch_1d_bkw_x_mpi);
    } else {
	/* Perform a set of forward 1D FFTs */
        fftw_execute(plan_batch_1d_fwd_x_mpi);
    }
    /* copy the results back to the input array. */
    dbl2dense(tmp_dbcom, afftbuf, nxfull, nrows, nzl);
    
    /*
     * Copy from afftbuf to ibuf. afftbuf is nxfull by nrows
     * obuf is nxl by nrows by mp_p
     */
    rows_from(afftbuf, obuf, chunksize_x, nrows, mp_p);
    /* 
     * The results of the FFT iare in
     * "alltoall order" as desired.
     */
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_p, chunksize_x, 0);
    }
    dense2guard(afftbuf, var, nxl, nyl, nzl);

    stop_omp_time();
    return(0);  /* return 0 to indicate success */
}


int prepBatchFFT(int lenFFTx, int numFFTx, int lenFFTy, int numFFTy,
                 complex double *var)
{
  /* This function  creates FFTW plans for forward and backward
     directions for x and for y.
     rank is set to 1 to perform 1D FFTs.
     Assume that the array will be transposed for the y-direction
     before calling the FFT function. Stride is thus one for
     both directions.
     numFFTx is normally nrows*nzl and "var" is a 3D array.
  */
  int len_x[]= {lenFFTx};
  int len_y[]= {lenFFTy};

  /*
  fftw_plan fftw_plan_many_dft(int rank, const int *n, int howmany,
                                  fftw_complex *in, const int *inembed,
                                  int istride, int idist,
                                  fftw_complex *out, const int *onembed,
                                  int ostride, int odist,
                                  int sign, unsigned flags);
  */
  plan_batch_1d_bkw_x= fftw_plan_many_dft(1, len_x, numFFTx, var, 0, 1, lenFFTx,
                                    var, 0, 1, lenFFTx, +1, FFTW_MEASURE);
  plan_batch_1d_fwd_x= fftw_plan_many_dft(1, len_x, numFFTx, var, 0, 1, lenFFTx,
                                    var, 0, 1, lenFFTx, -1, FFTW_MEASURE);
  plan_batch_1d_bkw_y= fftw_plan_many_dft(1, len_y, numFFTy, var, 0, 1, lenFFTy,
                                    var, 0, 1, lenFFTy, +1, FFTW_MEASURE);
  plan_batch_1d_fwd_y= fftw_plan_many_dft(1, len_y, numFFTy, var, 0, 1, lenFFTy,
                                    var, 0, 1, lenFFTy, -1, FFTW_MEASURE);
  ready_fft_batch= 1;
  return 0;
}

int prepBatchFFT_MPI(int lenFFTx, int numFFTx, int lenFFTy, int numFFTy,
                 complex double *var)
{
  /* This function  creates FFTW plans for forward and backward
     directions for x and for y.
     rank is set to 1 to perform 1D FFTs.
     Assume that the array will be transposed for the y-direction
     before calling the FFT function. Stride is thus one for
     both directions.
     numFFTx is normally nrows*nzl and "var" is a 3D array.
  */
  int len_x[]= {lenFFTx};
  int len_y[]= {lenFFTy};

  if(!mp_rank) printf("prepBatchFFT_loop: creating plans with lenFFTx=%d, numFFTX=%d, lenFFTy=%d, numFFTy=%d\n", lenFFTx, numFFTx, lenFFTy, numFFTy);

  plan_batch_1d_bkw_x_mpi= fftw_plan_many_dft(1, len_x, numFFTx, var, 0, 1, lenFFTx,
                                    var, 0, 1, lenFFTx, +1, FFTW_MEASURE);
  plan_batch_1d_fwd_x_mpi= fftw_plan_many_dft(1, len_x, numFFTx, var, 0, 1, lenFFTx,
                                    var, 0, 1, lenFFTx, -1, FFTW_MEASURE);
  plan_batch_1d_bkw_y_mpi= fftw_plan_many_dft(1, len_y, numFFTy, var, 0, 1, lenFFTy,
                                    var, 0, 1, lenFFTy, +1, FFTW_MEASURE);
  plan_batch_1d_fwd_y_mpi= fftw_plan_many_dft(1, len_y, numFFTy, var, 0, 1, lenFFTy,
                                    var, 0, 1, lenFFTy, -1, FFTW_MEASURE);
  ready_fft_batch_mpi= 1;
  return 0;
}

int prepFFT_loop(int lenFFTx, int numFFTx, int lenFFTy, int numFFTy)
{
  /* This function  creates FFTW plans for forward and backward
     directions for x and for y.
     rank is set to 1 to perform 1D FFTs. */

  if(!cmpvec_x) {
    cmpvec_x= (complex double *)malloc(lenFFTx*sizeof(complex double));
    if(!mp_rank) printf("prepFFT_loop: allocate cmpvec_x with %d elements\n", lenFFTx);
  }
  if(!cmpvec_y) {
    cmpvec_y= (complex double *)malloc(lenFFTy*sizeof(complex double));
    if(!mp_rank) printf("prepFFT_loop: allocate cmpvec_y_loc with %d elements\n", lenFFTy);
  }

  plan_loop_1d_bkw_x= fftw_plan_dft_1d(lenFFTx, cmpvec_x, cmpvec_x, +1, FFTW_MEASURE);
  plan_loop_1d_fwd_x= fftw_plan_dft_1d(lenFFTx, cmpvec_x, cmpvec_x, -1, FFTW_MEASURE);
  plan_loop_1d_bkw_y= fftw_plan_dft_1d(lenFFTy, cmpvec_y, cmpvec_y, +1, FFTW_MEASURE);
  plan_loop_1d_fwd_y= fftw_plan_dft_1d(lenFFTy, cmpvec_y, cmpvec_y, -1, FFTW_MEASURE);
  ready_fft_loop= 1;
  return 0;
}

int prepFFT_loop_loc(int lenFFTx, int numFFTx, int lenFFTy, int numFFTy)
{
  /* This function  creates FFTW plans for forward and backward
     directions for x and for y.
     rank is set to 1 to perform 1D FFTs. */

  if(!cmpvec_x_loc) {
    cmpvec_x_loc= (complex double *)malloc(lenFFTx*sizeof(complex double));
    if(!mp_rank) printf("prepFFT_loop_loc: allocate cmpvec_x_loc with %d elements\n", lenFFTx);
  }
  if(!cmpvec_y_loc) {
    cmpvec_y_loc= (complex double *)malloc(lenFFTy*sizeof(complex double));
    if(!mp_rank) printf("prepFFT_loop_loc: allocate cmpvec_y_loc with %d elements\n", lenFFTy);
  }

  plan_loop_1d_bkw_x_loc= fftw_plan_dft_1d(lenFFTx, cmpvec_x_loc, cmpvec_x_loc, +1, FFTW_MEASURE);
  plan_loop_1d_fwd_x_loc= fftw_plan_dft_1d(lenFFTx, cmpvec_x_loc, cmpvec_x_loc, -1, FFTW_MEASURE);
  plan_loop_1d_bkw_y_loc= fftw_plan_dft_1d(lenFFTy, cmpvec_y_loc, cmpvec_y_loc, +1, FFTW_MEASURE);
  plan_loop_1d_fwd_y_loc= fftw_plan_dft_1d(lenFFTy, cmpvec_y_loc, cmpvec_y_loc, -1, FFTW_MEASURE);
  ready_fft_loop_loc= 1;
  return 0;
}

int prepFFT_loop_full(int lenFFTx, int numFFTx, int lenFFTy, int numFFTy)
{
  /* This function  creates FFTW plans for forward and backward
     directions for x and for y.
     rank is set to 1 to perform 1D FFTs. */

  if(!cmpvec_x_full) {
    cmpvec_x_full= (complex double *)malloc(lenFFTx*sizeof(complex double));
    if(!mp_rank) printf("prepFFT_loop_full: allocate cmpvec_x_full with %d elements\n", lenFFTx);
  }
  if(!cmpvec_y_full) {
    cmpvec_y_full= (complex double *)malloc(lenFFTy*sizeof(complex double));
    if(!mp_rank) printf("prepFFT_loop_full: allocate cmpvec_y_full with %d elements\n", lenFFTy);
  }

  if(!mp_rank) printf("prepFFT_loop_full: Creating fftw scalar plans\n");
  plan_loop_1d_bkw_x_full= fftw_plan_dft_1d(lenFFTx, cmpvec_x_full, cmpvec_x_full, +1, FFTW_MEASURE);
  plan_loop_1d_fwd_x_full= fftw_plan_dft_1d(lenFFTx, cmpvec_x_full, cmpvec_x_full, -1, FFTW_MEASURE);
  plan_loop_1d_bkw_y_full= fftw_plan_dft_1d(lenFFTy, cmpvec_y_full, cmpvec_y_full, +1, FFTW_MEASURE);
  plan_loop_1d_fwd_y_full= fftw_plan_dft_1d(lenFFTy, cmpvec_y_full, cmpvec_y_full, -1, FFTW_MEASURE);
  ready_fft_loop_full= 1;
  return 0;
}

void fft_norm(rcomplex *grd, int numx, int numy, int numz, int lenFFTx, int lenFFTy)
{
  int ix, iy, iz;
  real fac;

  /* NOTE - FFT normalization is performed only after the FFT result
     is back in the pF3D array with guard zones.
  */
  fac = 1.0 / (lenFFTx * lenFFTy);
  for(iz= 0; iz < numz; iz++) {
    for(iy= 0; iy < numy; iy++) {
      for(ix= 0; ix < numx; ix++) {
        /* grd(ix,iy,iz) *= fac; */
        grd[ ((iz*(nyl+1))+iy)*(nxl+1)+ix ] *= fac;
      }
    }
  }
}


fftw_plan f3d_fftw3_create(double *x, double *xfft, long size, long dir)
{
  int direction;
  fftw_complex *in, *out;
  fftw_plan the_plan;

  out= (fftw_complex *)x;
  in= (fftw_complex *)xfft;

  /* Create the fftw plan data structure and return it.
     Convert from yorick style FFT directions to the convention
     used by FFTW. dir==1 means exp(ikx) which in FFTW
     is FFTW_BACKWARD. */
  if(dir == -1) direction= FFTW_FORWARD;
  else direction= FFTW_BACKWARD;

  the_plan= fftw_plan_dft_1d(size, in, out, direction, FFTW_MEASURE);
  return the_plan;
}
