#include <complex.h>
#include "fftw3.h"

#include "mytypes.h"
#include "pf3d_fft.h"
#include "grid.h"
#include "util.h"
#include "runparm.h"
#include "pf3dbench.h"

#include "pf3dbenchvars.h"

#if 0
extern rcomplex *afftbuf;
extern real *iptmp, *optmp;
extern double complex *tmp_dbcom;
#endif

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

#undef PRINT_VALS

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

void alltoall_chek_x(rcomplex *grd)
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

void alltoall_chek(rcomplex *grd)
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
      cmp_alltoallshift(afftbuf+izoff, obuf+izoff, mp_q, chunksize_y, 1);
    }
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_q, chunksize_y, 1);
    }
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(afftbuf+izoff, obuf+izoff, mp_p, chunksize_x, 0);
    }
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

void alltoall_chek_y(rcomplex *grd)
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
      cmp_alltoallshift(afftbuf+izoff, obuf+izoff, mp_q, chunksize_y, 1);
    }
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_q, chunksize_y, 1);
    }
#endif
    /* Put the data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void print_one(char *nam, rcomplex *buf, long ndx)
{
    rcomplex val;
    real rl, im;
  
    val= buf[ndx];
    rl= creal(val);
    im= cimag(val);
    if(!mp_rank) printf("%s[%ld]= %e + %e\n", nam, ndx, rl, im);
}

void trans_to(rcomplex *ibuf, rcomplex *obuf, int numx, int numy, int numz)
{
  int i, j, k;

  for(k=0; k<numz; k++) {
    for(i=0; i<numx; i++) {
      for(j=0; j<numy; j++) {
        obuf[j+i*numy+k*numx*numy]= ibuf[i+j*numx+k*numx*numy];
      }
    }
  }
}

void trans_from(rcomplex *ibuf, rcomplex *obuf, int numx, int numy, int numz)
{
  int i, j, k;

  /* transpose back from obuf to e */
  for(k=0; k<numz; k++) {
    for(j=0; j<numy; j++) {
      for(i=0; i<numx; i++) {
        obuf[i+j*numx+k*numx*numy] = ibuf[j+i*numy+k*numx*numy];
      }
    }
  }
}

void rows_to(rcomplex *ibuf, rcomplex *obuf, int chunksize, int nrows,
             int numP)
{
  int ip, i, j, iz;
  long bufoff, rowoff;
  
  /*
   * Copy from ibuf to obuf. ibuf is nxl by nrows by mp_p
   * obuf is nx by nrows
   */
  for(iz= 0; iz < nzl; iz++) {
    for(ip= 0; ip < numP; ip++) {
      for (j=0; j<nrows; j++) {
        bufoff= j*nxl+ip*chunksize+iz*nxl*nyl;
        rowoff= j*nxfull+ip*nxl+iz*nxl*nyl;
        for (i=0; i<nxl; i++) {
          obuf[i+rowoff] = ibuf[i+bufoff];
        }
      }
    }
  }
}

void rows_from(rcomplex *ibuf, rcomplex *obuf, int chunksize, int nrows,
               int numP)
{
  int ip, i, j, iz;
  long bufoff, rowoff;

  /*
   * Copy from ibuf to obuf. ibuf is nxfull by nrows
   * obuf is nxl by nrows by mp_p
   */
  for(iz= 0; iz < nzl; iz++) {
    for(ip= 0; ip < numP; ip++) {
      for (j=0; j<nrows; j++) {
        bufoff= j*nxl+ip*chunksize+iz*nxl*nyl;
        rowoff= j*nxfull+ip*nxl+iz*nxl*nyl;
        for (i=0; i<nxl; i++) {
          obuf[i+bufoff] = ibuf[i+rowoff];
        }
      }
    }
  }
}

void cols_to(rcomplex *ibuf, rcomplex *obuf, int chunksize, int ncols,
               int numQ)
{
  int iq, i, j, iz;
  long bufoff, coloff;

  /*
   * Copy from ibuf to obuf. ibuf is nyl by ncols by mp_q
   * obuf is nyfull by ncols
   */
  for(iz= 0; iz < nzl; iz++) {
    for(iq= 0; iq < numQ; iq++) {
      for (i=0; i<ncols; i++) {
        bufoff= i*nyl+iq*chunksize+iz*nxl*nyl;
        coloff= i*nyfull+iq*nyl+iz*nxl*nyl;
        for (j=0; j<nyl; j++) {
          obuf[j+coloff] = ibuf[j+bufoff];
        }
      }
    }
  }
}

void cols_from(rcomplex *ibuf, rcomplex *obuf, int chunksize, int ncols,
               int numQ)
{
  int iq, i, j, iz;
  long bufoff, coloff;

  /*
   * Copy from ibuf to obuf. ibuf is nyfull by ncols
   * obuf is nyl by ncols by mp_q
   */
  for(iz= 0; iz < nzl; iz++) {
    for(iq= 0; iq < numQ; iq++) {
      for (i=0; i<ncols; i++) {
        bufoff= i*nyl+iq*chunksize+iz*nxl*nyl;
        coloff= i*nyfull+iq*nyl+iz*nxl*nyl;
        for (j=0; j<nyl; j++) {
          obuf[j+bufoff] = ibuf[j+coloff];
        }
      }
    }
  }
}


void copy_x(rcomplex *grd)
{
    rcomplex *obuf = (rcomplex *)optmp;

    start_omp_time();

    /* Perform a 2D FFT. Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    rows_to(afftbuf, obuf, chunksize_x, nrows, mp_p);
    /* copy data to a complex double temporary array */
    dense2dbl(obuf, tmp_dbcom, nxfull, nrows, nzl);
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, obuf, nxfull, nrows, nzl);
    rows_from(obuf, afftbuf, chunksize_x, nrows, mp_p);

    /* Put the data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void copy_y(rcomplex *grd)
{
    rcomplex *obuf = (rcomplex *)optmp;
    int nelem= nxl*nyl*nzl;

    start_omp_time();

    /* Perform a 2D FFT. Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);

    /* Transpose to a "y varies fastest" checkerboard. */
    trans_to(afftbuf, obuf, nxl, nyl, nzl);
    cols_to(obuf, afftbuf, chunksize_y, ncols, mp_q);
    /* copy data to a complex double temporary array */
    dense2dbl(afftbuf, tmp_dbcom, nyfull, ncols, nzl);
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nyfull, ncols, nzl);
    cols_from(afftbuf, obuf, chunksize_y, ncols, mp_q);
    /* Transpose to a "x varies fastest" checkerboard. */
    trans_from(obuf, afftbuf, nxl, nyl, nzl);

    /* Put the data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void copy_xy(rcomplex *grd)
{
    rcomplex *obuf = (rcomplex *)optmp;

    start_omp_time();

    /* Perform a 2D FFT. Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    rows_to(afftbuf, obuf, chunksize_x, nrows, mp_p);
    /* copy data to a complex double temporary array */
    dense2dbl(obuf, tmp_dbcom, nxfull, nrows, nzl);
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, obuf, nxfull, nrows, nzl);
    rows_from(obuf, afftbuf, chunksize_x, nrows, mp_p);

    trans_to(afftbuf, obuf, nxl, nyl, nzl);
    cols_to(obuf, afftbuf, chunksize_y, ncols, mp_q);
    /* copy data to a complex double temporary array */
    dense2dbl(afftbuf, tmp_dbcom, nyfull, ncols, nzl);
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nyfull, ncols, nzl);
    cols_from(afftbuf, obuf, chunksize_y, ncols, mp_q);
    /* Transpose to a "x varies fastest" checkerboard. */
    trans_from(obuf, afftbuf, nxl, nyl, nzl);
    /* Put the data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void chek_dbl(rcomplex *grd)
{
    start_omp_time();

    /* Perform a 2D FFT. Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    dense2dbl(afftbuf, tmp_dbcom, nyfull, ncols, nzl);
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nyfull, ncols, nzl);
    /* Put the data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void chek_rows(rcomplex *grd)
{
    rcomplex *obuf = (rcomplex *)optmp;

    start_omp_time();

    /* Perform a 2D FFT. Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    rows_to(afftbuf, obuf, chunksize_x, nrows, mp_p);
    rows_from(obuf, afftbuf, chunksize_x, nrows, mp_p);
    /* Put the data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void chek_cols(rcomplex *grd)
{
    rcomplex *obuf = (rcomplex *)optmp;

    start_omp_time();

    /* Perform a 2D FFT. Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    trans_to(afftbuf, obuf, nxl, nyl, nzl);
    cols_to(obuf, afftbuf, chunksize_y, ncols, mp_q);
    cols_from(afftbuf, obuf, chunksize_y, ncols, mp_q);
    /* Transpose to a "x varies fastest" checkerboard. */
    trans_from(obuf, afftbuf, nxl, nyl, nzl);
    /* Put the data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void chek_trans(rcomplex *grd)
{
    rcomplex *obuf = (rcomplex *)optmp;

    start_omp_time();

    /* Perform a 2D FFT. Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    trans_to(afftbuf, obuf, nxl, nyl, nzl);
    /* Transpose to a "x varies fastest" checkerboard. */
    trans_from(obuf, afftbuf, nxl, nyl, nzl);
    /* Put the data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void move_2D(rcomplex *grd)
{
    rcomplex *obuf = (rcomplex *)optmp;
    int iz;
    long izoff;

    start_omp_time();

    /* Perform all data movement required for a 2D FFT. Includes
       MPI message passing if an xy-plane
       is decomposed into multiple domains. */
#ifndef USE_MPI
    /* remove the guard zones from the input array */
    guard2dense(grd, obuf, nxl, nyl, nzl);
    puts("WARNING - move_2d only works in an MPI code");
#else
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(afftbuf+izoff, obuf+izoff, mp_p, chunksize_x, 0);
    }
#endif
    rows_to(obuf, afftbuf, chunksize_x, nrows, mp_p);
    /* copy data to a complex double temporary array */
    dense2dbl(afftbuf, tmp_dbcom, nxfull, nrows, nzl);
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nxfull, nrows, nzl);
    rows_from(afftbuf, obuf, chunksize_x, nrows, mp_p);
#ifndef USE_MPI
    puts("WARNING - move_2d only works in an MPI code");
#else
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_p, chunksize_x, 0);
    }
#endif
    
    /* Transpose to a "y varies fastest" checkerboard. */
    trans_to(afftbuf, obuf, nxl, nyl, nzl);
#ifndef USE_MPI
    puts("WARNING - move_2d only works in an MPI code");
#else
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_q, chunksize_y, 1);
    }
#endif
    cols_to(afftbuf, obuf, chunksize_y, ncols, mp_q);
    /* copy data to a complex double temporary array */
    dense2dbl(obuf, tmp_dbcom, nyfull, ncols, nzl);
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, obuf, nyfull, ncols, nzl);
    cols_from(obuf, afftbuf, chunksize_y, ncols, mp_q);
#ifndef USE_MPI
    puts("WARNING - move_2d only works in an MPI code");
#else
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(afftbuf+izoff, obuf+izoff, mp_q, chunksize_y, 1);
    }
#endif
    /* Transpose to a "x varies fastest" checkerboard. */
    trans_from(obuf, afftbuf, nxl, nyl, nzl);
    /* Put the un-normalized FFT data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void move_1D(rcomplex *grd)
{
    rcomplex *obuf = (rcomplex *)optmp;
    int iz;
    long izoff;

    start_omp_time();

    /* Perform data motion for the first half of a 2D FFT.
       Includes MPI message passing if an xy-plane
       is decomposed into multiple domains. */
    /* remove the guard zones from the input array */
    guard2dense(grd, afftbuf, nxl, nyl, nzl);
#ifndef USE_MPI
    puts("WARNING - move_1d only works in an MPI code");
#else
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(afftbuf+izoff, obuf+izoff, mp_p, chunksize_x, 0);
    }
#endif
    rows_to(obuf, afftbuf, chunksize_x, nrows, mp_p);
    /* copy data to a complex double temporary array */
    dense2dbl(afftbuf, tmp_dbcom, nxfull, nrows, nzl);
    /* copy the results back to an rcomplex array */
    dbl2dense(tmp_dbcom, afftbuf, nxfull, nrows, nzl);
    rows_from(afftbuf, obuf, chunksize_x, nrows, mp_p);
#ifndef USE_MPI
    puts("WARNING - move_1d only works in an MPI code");
#else
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obuf+izoff, afftbuf+izoff, mp_p, chunksize_x, 0);
    }
#endif
    
    /* Put the un-normalized FFT data back in the original array */
    dense2guard(afftbuf, grd, nxl, nyl, nzl);
    stop_omp_time();
}

void guard2guard(rcomplex *grd, rcomplex *grd2, int nxl, int nyl, int nzl)
{
  int ix, iy, iz;

  /* grd must have guard zones like t0 and other light waves.
 *      dns has no guard zones.
 *           NOTE - this function is always called for arrays where
 *                the x-index varies fastest (i.e. normal pF3D order).
 *                     Array dimensions are the normal nxl-by-nyl-by-nzl. */
  for(iz= 0; iz < nzl; iz++) {
    for(iy= 0; iy < nyl; iy++) {
      for(ix= 0; ix < nxl; ix++) {
        grd2(ix,iy,iz)= grd(ix,iy,iz);
      }
    }
  }
}

void guard2dense(rcomplex *grd, rcomplex *dns, int nxl, int nyl, int nzl)
{
  int ix, iy, iz;

  /* grd must have guard zones like t0 and other light waves.
     dns has no guard zones.
     NOTE - this function is always called for arrays where
     the x-index varies fastest (i.e. normal pF3D order).
     Array dimensions are the normal nxl-by-nyl-by-nzl. */
  for(iz= 0; iz < nzl; iz++) {
    for(iy= 0; iy < nyl; iy++) {
      for(ix= 0; ix < nxl; ix++) {
        dns(ix,iy,iz)= grd(ix,iy,iz);
      }
    }
  }
}

void dense2guard(rcomplex *dns, rcomplex *grd, int nxl, int nyl, int nzl)
{
  int ix, iy, iz;

  /* grd must have guard zones like t0 and other light waves.
     dns has no guard zones.
     NOTE - this function is always called for arrays where
     the x-index varies fastest (i.e. normal pF3D order).
     Array dimensions are the normal nxl-by-nyl-by-nzl. */
  for(iz= 0; iz < nzl; iz++) {
    for(iy= 0; iy < nyl; iy++) {
      for(ix= 0; ix < nxl; ix++) {
        grd(ix,iy,iz)= dns(ix,iy,iz);
      }
    }
  }
}

void dense2dbl(rcomplex *dns, complex double *dbl, int num_i, int num_j,
               int num_k)
{
  long i, nvals;

  nvals= num_i*(long)num_j*num_k;
  
  /* dns and dbl have no guard zones. dbl is definitely
     "complex double". dns is usually "complex float" but sometimes
     is "complex double".
     WARNING - this function might operate on nxl-by-nyl-by-nzl arrays,
     nxfull-by-nrows-by-nzl arrays, or nyfull-by-ncols-by-nzl arrays.
     Because the array is dense, compute the length and treat as 1D. */
  for(i= 0; i < nvals; i++) {
    dbl[i]= dns[i];
  }
}

void dbl2dense(complex double *dbl, rcomplex *dns, int num_i, int num_j,
               int num_k)
{
  long i, nvals;

  nvals= num_i*(long)num_j*num_k;

  /* dns and dbl have no guard zones. dbl is definitely
     "complex double". dns is usually "complex float" but sometimes
     is "complex double".
     WARNING - this function might operate on nxl-by-nyl-by-nzl arrays,
     nxfull-by-nrows-by-nzl arrays, or nyfull-by-ncols-by-nzl arrays.
     Because the array is dense, compute the length and treat as 1D. */
  for(i= 0; i < nvals; i++) {
    dns[i]= dbl[i];
  }
}

void test_alltoall(rcomplex * restrict var, int nxfull, int nyfull, int nzl)
{
    int        ix, iy, ndx, nrows, chunksize;
    rcomplex  *obuf = (rcomplex *)optmp;
    rcomplex  *tbuf;

    /* Synchronize all MPI ranks) */
#ifdef USE_MPI
    simsync();
#endif
    printf("test_alltoall called with nxfull=%d, nyfull=%d, nzl=%d, var=%p\n", nxfull, nyfull, nzl, var);

    chunksize= nxl*nrows;

    /* An MPI_Alltoall will be used to move the data into obuf.
       The data in "var" is already in the proper order, so use "var"
       as the "input buffer" for the message passing. */
#ifdef USE_MPI
    cmp_alltoallshift(var, obuf, g3.Q, sizeof(rcomplex)/sizeof(real)*chunksize, 0);
#endif

    /*
     * Copy the results into var
     */
    ndx= 0;
    tbuf= (rcomplex *)obuf;
    for(iy= 0; iy < nyfull; iy++) {
        for (ix=0; ix < nxfull; ix++) {
          var(ix,iy) = tbuf[ndx];
          ndx++;
        }
    }
#ifdef USE_MPI
    simsync();
#endif
}

void prep_alltoall(rcomplex * restrict var, int nxfull, int nyfull, int nzl)
{
    int        ix, iy, ndx;

    /* Initialize var to increasing integers with a different
       offset for each MPI rank */
    for(iy= 0; iy < nyfull; iy++) {
        for (ix=0; ix < nxfull; ix++) {
#ifdef USE_MPI
          var(ix,iy) = ix+nxfull*iy+1024*1024*g3.me;
#else
          var(ix,iy) = ix+nxfull*iy;
#endif
        }
    }
}
