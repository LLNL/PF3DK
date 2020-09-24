/* pf3d_fft.h */

#include <stdlib.h>
#include <math.h>
#include "mytypes.h"

/* FFTs in pf3d
   
   pf3d makes heavy use of parallel 2D FFTs.
   The 2D FFT consists of two parts. the first step is a message passing
   phase that transposes from pf3d's "checkerboard" XY decomposition to
   a decomposition in which indvidual processes have complete rows or
   columns of an XY plane. Processors then apply a normal 1D FFT and
   transpose back.
*/

/* debug printing function */
int dprint(char *rtn, char *fmt, ...);

/* These are utility functions to copy between arrays with
   guard zones and arrays withou, copy between complex float and
   complex double arrays, etc. */
void fft_norm(rcomplex *grd, int numx, int numy, int numz,
              int lenFFTx, int lenFFTy);
void guard2dense(rcomplex *grd, rcomplex *dns, int nxl, int nyl, int nzl);
void dense2guard(rcomplex *dns, rcomplex *grd, int nxl, int nyl, int nzl);
void dense2dbl(rcomplex *dns, complex double *dbl, int num_i, int num_j,
               int num_k);
void dbl2dense(complex double *dbl, rcomplex *dns, int num_i, int num_j,
               int num_k);
void rows_to(rcomplex *ibuf, rcomplex *obuf, int chunksize, int nrows,
             int numP);
void rows_from(rcomplex *ibuf, rcomplex *obuf, int chunksize, int nrows,
               int numP);
void cols_to(rcomplex *ibuf, rcomplex *obuf, int chunksize, int ncols,
             int numQ);
void cols_from(rcomplex *ibuf, rcomplex *obuf, int chunksize, int ncols,
               int numQ);
void trans_to(rcomplex *ibuf, rcomplex *obuf, int numx, int numy, int numz);
void trans_from(rcomplex *ibuf, rcomplex *obuf, int numx, int numy, int numz);

void rcmp2dev(rcomplex *var, rcomplex *d_var, int num);
void realPrefetch(real *var, int num);
void rcmpPrefetch(rcomplex *var, int num);
void dcmpPrefetch(dcomplex *var, int num);
void fft_norm_omp45(rcomplex *grd, int numx, int numy, int numz,
                    int lenFFTx, int lenFFTy);
void guard2dense_omp45(rcomplex *grd, rcomplex *dns, int nxl, int nyl, int nzl);
void dense2guard_omp45(rcomplex *dns, rcomplex *grd, int nxl, int nyl, int nzl);
void dense2dbl_omp45(rcomplex *dns, dcomplex *dbl, int num_i, int num_j,
               int num_k);
void dbl2dense_omp45(dcomplex *dbl, rcomplex *dns, int num_i, int num_j,
               int num_k);
void rows_to_omp45(rcomplex *ibuf, rcomplex *obuf, int chunksize, int nrows,
             int numP);
void rows_from_omp45(rcomplex *ibuf, rcomplex *obuf, int chunksize, int nrows,
               int numP);
void cols_to_omp45(rcomplex *ibuf, rcomplex *obuf, int chunksize, int ncols,
             int numQ);
void cols_from_omp45(rcomplex *ibuf, rcomplex *obuf, int chunksize, int ncols,
               int numQ);
void trans_to_omp45(rcomplex *ibuf, rcomplex *obuf, int numx, int numy, int numz);
void trans_from_omp45(rcomplex *ibuf, rcomplex *obuf, int numx, int numy, int numz);
void mak_dev_dcomplex(int num, void **ptr);
void mak_dev_rcomplex(int num, void **ptr);
void init_gpubuf(long nzon);
void test_alltoall(rcomplex * restrict var, int nxfull, int nyfull, int nzl);
void prep_alltoall(rcomplex * restrict var, int nxfull, int nyfull, int nzl);

int fft_1D_batch_x(rcomplex *var, int lenFFT, int numFFT, int isign, int nzl);
int fft_1D_batch_y(rcomplex *var, int lenFFT, int numFFT, int isign, int nzl);
int fft_1D_batch_mpi_x(rcomplex *var, int lenFFT, int numFFT, int isign,
                       int nxl, int nyl, int nzl);
int fft_1D_loop_x(complex double *var, int lenFFT, int numFFT,
                  int isign, int iz);
int fft_1D_loop_y(complex double *var, int lenFFT, int numFFT,
                  int isign, int iz);
int fft_1D_loop_x_loc(complex double *var, int lenFFT, int numFFT,
                      int isign, int iz);
int fft_1D_loop_y_loc(complex double *var, int lenFFT, int numFFT,
                      int isign, int iz);
int fft_1D_loop_x_full(complex double *var, int lenFFT, int numFFT, int isign,
                       int iz);
int fft_1D_loop_y_full(complex double *var, int lenFFT, int numFFT, int isign,
                       int iz);
void FFT_2d_many(rcomplex *grd, int isign);
void FFT1D_x(rcomplex *grd, int isign);
void FFT2D(rcomplex *grd, int isign);
void FFT2D_noguard(rcomplex *grd, int isign);
void FFT_chek_x_loc(rcomplex *grd, int isign);
void FFT_chek_y_loc(rcomplex *grd, int isign);
void FFT_chek_loc(rcomplex *grd, int isign);
void FFT_chek_x(rcomplex *grd, int isign);
void FFT_chek_y(rcomplex *grd, int isign);
int prepBatchFFT(int lenFFTx, int numFFTx, int lenFFTy, int numFFTy,
                 complex double *var);
int prepBatchFFT_MPI(int lenFFTx, int numFFTx, int lenFFTy, int numFFTy,
                 complex double *var);
int prepFFT_loop(int lenFFTx, int numFFTx, int lenFFTy, int numFFTy);
int prepFFT_loop_loc(int lenFFTx, int numFFTx, int lenFFTy, int numFFTy);
int prepFFT_loop_full(int lenFFTx, int numFFTx, int lenFFTy, int numFFTy);
void copy_x(rcomplex *grd);
void copy_y(rcomplex *grd);
void copy_xy(rcomplex *grd);
void chek_rows(rcomplex *grd);
void chek_cols(rcomplex *grd);
void chek_trans(rcomplex *grd);
void move_2D(rcomplex *grd);
void move_1D(rcomplex *grd);
void alltoall_chek_x(rcomplex *grd);
void alltoall_chek_y(rcomplex *grd);
void alltoall_chek(rcomplex *grd);
void chek_dbl(rcomplex *grd);


/* Functions that use cufft from Nvidia */
void fft_1d_cufft(rcomplex *e, int isign, int nxl, int nyl, int nzl);
void fft_2d_cufft(rcomplex *e, int nxfull, int nyfull, int isign,
                  int nxl, int nyl, int nzl);
void fft_2d_cufft_mpi(rcomplex *e, int nxfull, int nyfull, int isign,
                      int nxl, int nyl, int nzl);
void fft_1d_cufft_msg(rcomplex *e, int nxfull, int isign,
                      int nxl, int nyl, int nzl);
