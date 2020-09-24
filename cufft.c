#include <cufft.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include "mytypes.h"
#include "runparm.h"
#include "pf3dbench.h"
#include "pf3d_fft.h"
#include "grid.h"
#include "load_data.h"
#include "nvToolsExt.h"

#include "pf3dbenchvars.h"

extern double complex *tmp_dbcom;

cufftHandle plan_many_x_mpi= 0;
cufftHandle plan_many_y_mpi= 0;
cufftHandle plan_many_x_msg= 0;
cufftHandle plan_many_y_msg= 0;
cufftHandle plan_many_x_loc= 0;
cufftHandle plan_many_y_loc= 0;
cufftHandle plan_1d_x_loc= 0;

#define e(i,j,k) e[(((k)*(nyl+1) + (j))*(nxl+1) + (i))]
#define tmp_dbcom(a,b,c) tmp_dbcom[a+b*nxl+c*nxl*nyl]
#define tmp_ydbcom(a,b,c) tmp_dbcom[b+a*nyl+c*nxl*nyl]

cufftDoubleComplex *devDptr= 0;
cufftComplex *obufDevptr= 0;
cufftComplex *ibufDevptr= 0;

void mak_dev_dcomplex(int num, void **ptr)
{
  cudaMallocManaged(ptr, num*sizeof(cufftDoubleComplex),
                    cudaMemAttachGlobal);
  /* Force the data to the GPU. It will be random bytes at this time. */
  dcmpPrefetch(*ptr, ngtot);
}

void mak_dev_rcomplex(int num, void **ptr)
{
  cudaMallocManaged(ptr, num*sizeof(cufftComplex), cudaMemAttachGlobal);
  /* Force the data to the GPU. It will be random bytes at this time. */
  rcmpPrefetch(*ptr, ngtot);
}

void init_gpubuf(long nzon)
{
  /* initialize buffers for use in 2D FFTs on the GPU.
     t0DevPtr is used when the host wants to copy
     t0 into GPU memory. */
  mak_dev_dcomplex(nzon, (void **)&devDptr);
  mak_dev_rcomplex(nzon, (void **)&obufDevptr);
  mak_dev_rcomplex(nzon, (void **)&ibufDevptr);
  mak_dev_rcomplex(nzon, (void **)&t0DevPtr);
}

void fft_1d_cufft(rcomplex *e, int isign, int nxl, int nyl, int nzl)
{
    cufftResult res;
    int izoff, iz;
    long nzon;
    rcomplex *obufDev, *ibufDev;
    dcomplex *tvarDblDev;

    nzon= nxl*nyl*nzl;
    if(!devDptr) mak_dev_dcomplex(nzon, (void **)&devDptr);
    if(!obufDevptr) mak_dev_rcomplex(nzon, (void **)&obufDevptr);
    if(!ibufDevptr) mak_dev_rcomplex(nzon, (void **)&ibufDevptr);
    /* Nvidia insists that complex has to be a struct.
       Set some pointers so bufDevfers can be used as C99 complex. */
    tvarDblDev= (dcomplex *)devDptr;
    obufDev= (rcomplex *)obufDevptr;
    ibufDev= (rcomplex *)ibufDevptr;
    /*
      cufftResult 
        cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed,
          int istride, int idist, int *onembed, int ostride,
          int odist, cufftType type, int batch);
      cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);
      cufftExecC2C(plan, data, data, CUFFT_FORWARD);
     */
    if(!plan_1d_x_loc) {
      cufftPlan1d(&plan_1d_x_loc, nxl, CUFFT_Z2Z, nyl*nzl);
    }
    
    /* Copy data from e into afftbufDev while removing guard zones.
       Exchange messages with other MPI ranks using MPI_Alltoall.
       Transpose from "alltoall order" to rows.
       Copy into a double complex array.
       Perform 1D FFTs in the x-direction.
       Copy back to "real" precision.
       Transpose from rows to "alltoall order".
       Exchange messages with other MPI ranks using MPI_Alltoall.
    */

    guard2dense_omp45(e, obufDev , nxl, nyl, nzl);
    /* copy data to the array in unified memory */
    dense2dbl_omp45(obufDev, tvarDblDev, nxl, nyl, nzl);
    
    /*.................... Batched 1d FFT in X-direction */
    if(isign == 1) {
        cufftExecZ2Z(plan_1d_x_loc, devDptr, devDptr, CUFFT_INVERSE);
    } else {
        cufftExecZ2Z(plan_1d_x_loc, devDptr, devDptr, CUFFT_FORWARD);
    }
    cudaDeviceSynchronize();
    /* copy data to the CPU array */
    dbl2dense_omp45(tvarDblDev, obufDev, nxl, nyl, nzl);

    /* put back in array with guard zones */
    dense2guard_omp45(obufDev, e, nxl, nyl, nzl);
}


void fft_2d_cufft(rcomplex *e, int nxfull, int nyfull, int isign,
                  int nxl, int nyl, int nzl)
{
    cufftResult res;
    int izoff, iz;
    long nzon;
    rcomplex *obufDev, *ibufDev;
    dcomplex *tvarDblDev;

    nzon= nxl*nyl*nzl;
    if(!devDptr) mak_dev_dcomplex(nzon, (void **)&devDptr);
    if(!obufDevptr) mak_dev_rcomplex(nzon, (void **)&obufDevptr);
    if(!ibufDevptr) mak_dev_rcomplex(nzon, (void **)&ibufDevptr);
    /* Nvidia insists that complex has to be a struct.
       Set some pointers so bufDevfers can be used as C99 complex. */
    tvarDblDev= (dcomplex *)devDptr;
    obufDev= (rcomplex *)obufDevptr;
    ibufDev= (rcomplex *)ibufDevptr;
    /*
      cufftResult 
        cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed,
          int istride, int idist, int *onembed, int ostride,
          int odist, cufftType type, int batch);
      cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH);
      cufftExecC2C(plan, data, data, CUFFT_FORWARD);
     */
    if(!plan_many_x_loc) {
      cufftPlan1d(&plan_many_x_loc, nxl, CUFFT_Z2Z, nyl*nzl);
    }
    if(!plan_many_y_loc) {
      cufftPlan1d(&plan_many_y_loc, nyl, CUFFT_Z2Z, nxl*nzl);
    }
    
    /***********************************************************************
     ****                  Part 1 -- X Direction sweep                  ****
     ***********************************************************************/
    /* Copy data from e into afftbufDev while removing guard zones.
       Exchange messages with other MPI ranks using MPI_Alltoall.
       Transpose from "alltoall order" to rows.
       Copy into a double complex array.
    */
    guard2dense_omp45(e, obufDev, nxl, nyl, nzl);
    /* copy data to the array in unified memory */
    dense2dbl_omp45(obufDev, tvarDblDev, nxl, nyl, nzl);

    /*.................... Batched 1d FFT in X-direction */
    if(isign == 1) {
        cufftExecZ2Z(plan_many_x_loc, devDptr, devDptr, CUFFT_INVERSE);
    } else {
        cufftExecZ2Z(plan_many_x_loc, devDptr, devDptr, CUFFT_FORWARD);
    }
    cudaDeviceSynchronize();
    /* copy data to the CPU array */
    dbl2dense_omp45(tvarDblDev, obufDev, nxl, nyl, nzl);

    /***********************************************************************
     ****                  Part 2 -- Y Direction sweep                  ****
     ***********************************************************************/
    /* Transpose from row-major to column-major.
       Exchange messages with other MPI ranks using MPI_Alltoall.
       Transpose from "alltoall order" to columns.
       Copy into a double complex array.
       Perform 1D FFTs in the x-direction.
       Copy back to "real" precision.
       Transpose from columns to "alltoall order".
       Exchange messages with other MPI ranks using MPI_Alltoall.
       Copy back into an array with guard zones.
    */

    trans_to_omp45(obufDev, ibufDev, nxl, nyl, nzl);
    dense2dbl_omp45(ibufDev, tvarDblDev, nyl, nxl, nzl);

    /*.................... Batched 1d FFT in X-direction */
    if(isign == 1) {
        cufftExecZ2Z(plan_many_y_loc, devDptr, devDptr, CUFFT_INVERSE);
    } else {
        cufftExecZ2Z(plan_many_y_loc, devDptr, devDptr, CUFFT_FORWARD);
    }
    cudaDeviceSynchronize();
    /* copy data to the CPU array */
    dbl2dense_omp45(tvarDblDev, obufDev, nxl, nyl, nzl);

    trans_from_omp45(obufDev, ibufDev, nxl, nyl, nzl);
    /* remove gaurd zones */
    dense2guard_omp45(ibufDev, e, nxl, nyl, nzl);
}

void fft_2d_cufft_mpi(rcomplex *e, int nxfull, int nyfull, int isign,
                  int nxl, int nyl, int nzl)
{
    cufftResult res;
    int izoff, iz;
    long nzon;
    rcomplex *obufDev, *ibufDev;
    dcomplex *tvarDblDev;

    nvtxRangePushA("2d_cufft_mpi_prep");
    nzon= nxl*nyl*nzl;
    if(!devDptr) mak_dev_dcomplex(nzon, (void **)&devDptr);
    if(!obufDevptr) mak_dev_rcomplex(nzon, (void **)&obufDevptr);
    if(!ibufDevptr) mak_dev_rcomplex(nzon, (void **)&ibufDevptr);
    /* Nvidia insists that complex has to be a struct.
       Set some pointers so bufDevfers can be used as C99 complex. */
    tvarDblDev= (dcomplex *)devDptr;
    obufDev= (rcomplex *)obufDevptr;
    ibufDev= (rcomplex *)ibufDevptr;

    if(!plan_many_x_mpi) {
      cufftPlan1d(&plan_many_x_mpi, nxfull, CUFFT_Z2Z, nrows*nzl);
    }
    if(!plan_many_y_mpi) {
      cufftPlan1d(&plan_many_y_mpi, nyfull, CUFFT_Z2Z, ncols*nzl);
    }
    nvtxRangePop();
    
    /***********************************************************************
     ****                  Part 1 -- X Direction sweep                  ****
     ***********************************************************************/
    /* Copy data from e into afftbufDev while removing guard zones.
       Exchange messages with other MPI ranks using MPI_Alltoall.
       Transpose from "alltoall order" to rows.
       Copy into a double complex array.
       Perform 1D FFTs in the x-direction.
       Copy back to "real" precision.
       Transpose from rows to "alltoall order".
       Exchange messages with other MPI ranks using MPI_Alltoall.
    */

    nvtxRangePushA("2d_cufft_mpi_guard2dense");
    guard2dense_omp45(e, ibufDev , nxl, nyl, nzl);
    nvtxRangePop();
    nvtxRangePushA("2d_cufft_mpi_alltoall");
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(ibufDev+izoff, obufDev+izoff, mp_p, chunksize_x, 0);
    }
    nvtxRangePop();
    /*
     * obufDev is nxl by nrows by mp_p
     * copy into afftbufDev which is nxfull by nrows
     */
    nvtxRangePushA("2d_cufft_mpi_rows_to");
    rows_to_omp45(obufDev, ibufDev, chunksize_x, nrows, mp_p);
    /* copy data to the array in unified memory */
    dense2dbl_omp45(ibufDev, tvarDblDev, nxfull, nrows, nzl);
    nvtxRangePop();

    nvtxRangePushA("2d_cufft_mpi_cufft_x");
    /*.................... Batched 1d FFT in X-direction */
    if(isign == 1) {
        cufftExecZ2Z(plan_many_x_mpi, devDptr, devDptr, CUFFT_INVERSE);
    } else {
        cufftExecZ2Z(plan_many_x_mpi, devDptr, devDptr, CUFFT_FORWARD);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
    nvtxRangePushA("2d_cufft_mpi_rows_from");
    /* copy data to the CPU array */
    dbl2dense_omp45(tvarDblDev, ibufDev, nxfull, nrows, nzl);

    /*
     * Copy from afftbufDev to obufDev. afftbufDev is nxfull by nrows
     * obufDev is nxl by nrows by mp_p
     */
    rows_from_omp45(ibufDev, obufDev, chunksize_x, nrows, mp_p);
    nvtxRangePop();
    /* 
     * The results of the FFT iare in
     * "alltoall order" as desired.
     */
    nvtxRangePushA("2d_cufft_mpi_alltoall2");
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obufDev+izoff, ibufDev+izoff, mp_p, chunksize_x, 0);
    }
    nvtxRangePop();

    /***********************************************************************
     ****                  Part 2 -- Y Direction sweep                  ****
     ***********************************************************************/
    /* Transpose from row-major to column-major.
       Exchange messages with other MPI ranks using MPI_Alltoall.
       Transpose from "alltoall order" to columns.
       Copy into a double complex array.
       Perform 1D FFTs in the x-direction.
       Copy back to "real" precision.
       Transpose from columns to "alltoall order".
       Exchange messages with other MPI ranks using MPI_Alltoall.
       Copy back into an array with guard zones.
    */

    nvtxRangePushA("2d_cufft_mpi_trans_to");
    trans_to_omp45(ibufDev, obufDev, nxl, nyl, nzl);
    nvtxRangePop();
    nvtxRangePushA("2d_cufft_mpi_alltoall3");
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose to a row based decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(obufDev+izoff, ibufDev+izoff, mp_q, chunksize_y, 1);
    }
    nvtxRangePop();
    /*
     * obufDev is nyl by ncols by mp_q
     * copy into afftbufDev which is nyfull by ncols
     */
    nvtxRangePushA("2d_cufft_mpi_cols_to");
    cols_to_omp45(ibufDev, obufDev, chunksize_y, ncols, mp_q);
    /* copy data to the array in unified memory */
    dense2dbl_omp45(obufDev, tvarDblDev, nyfull, ncols, nzl);
    nvtxRangePop();

    nvtxRangePushA("2d_cufft_mpi_cufft_y");
    /*.................... Batched 1d FFT in X-direction */
    if(isign == 1) {
        cufftExecZ2Z(plan_many_y_mpi, devDptr, devDptr, CUFFT_INVERSE);
    } else {
        cufftExecZ2Z(plan_many_y_mpi, devDptr, devDptr, CUFFT_FORWARD);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
    /* copy the results back to the input array. */
    nvtxRangePushA("2d_cufft_mpi_cols_from");
    dbl2dense_omp45(tvarDblDev, obufDev, nyfull, ncols, nzl);
    /*
     * perform a reverse copy from afftbufDev to ibufDev
     */
    cols_from_omp45(obufDev, ibufDev, chunksize_y, ncols, mp_q);
    nvtxRangePop();
    nvtxRangePushA("2d_cufft_mpi_alltoall4");
    for(iz= 0; iz < nzl; iz++) {
      izoff= iz*nxl*nyl;
      /* Use MPI_Alltoall to transpose back to a checkerboard decomposition.
         Pass messages for one xy-plane at a time. */
      cmp_alltoallshift(ibufDev+izoff, obufDev+izoff, mp_q, chunksize_y, 1);
    }
    nvtxRangePop();

    /* transpose back from obufDev to e */
    nvtxRangePushA("2d_cufft_mpi_trans_from");
    trans_from_omp45(obufDev, ibufDev, nxl, nyl, nzl);
    dense2guard_omp45(ibufDev, e, nxl, nyl, nzl);
    nvtxRangePop();
}




void fft_1d_cufft_msg(rcomplex *e, int nxfull, int isign,
                  int nxl, int nyl, int nzl)
{
    cufftResult res;
    int izoff, iz;
    long nzon;
    rcomplex *obufDev, *ibufDev;
    dcomplex *tvarDblDev;

    nvtxRangePushA("1d_cufft_msg_prep");
    nzon= nxl*nyl*nzl;
    if(!devDptr) mak_dev_dcomplex(nzon, (void **)&devDptr);
    if(!obufDevptr) mak_dev_rcomplex(nzon, (void **)&obufDevptr);
    if(!ibufDevptr) mak_dev_rcomplex(nzon, (void **)&ibufDevptr);
    /* Nvidia insists that complex has to be a struct.
       Set some pointers so bufDevfers can be used as C99 complex. */
    tvarDblDev= (dcomplex *)devDptr;
    obufDev= (rcomplex *)obufDevptr;
    ibufDev= (rcomplex *)ibufDevptr;

    if(!plan_many_x_msg) {
      cufftPlan1d(&plan_many_x_mpi, nxfull, CUFFT_Z2Z, nrows*nzl);
    }
    nvtxRangePop();
    
    /***********************************************************************
     ****                  Part 1 -- X Direction sweep                  ****
     ***********************************************************************/
    /* Copy data from e into afftbufDev while removing guard zones.
       Exchange messages with other MPI ranks using MPI_Alltoall.
       Transpose from "alltoall order" to rows.
       Copy into a double complex array.
       Perform 1D FFTs in the x-direction.
       Copy back to "real" precision.
       Transpose from rows to "alltoall order".
       Exchange messages with other MPI ranks using MPI_Alltoall.
    */

    nvtxRangePushA("1d_cufft_msg_guard2dense");
    guard2dense_omp45(e, ibufDev , nxl, nyl, nzl);
    nvtxRangePop();
    nvtxRangePushA("1d_cufft_msg_isend");
    /* Use MPI_Isend to send half of all partial rows
       to the other MPI process in the same row of the decomp. */
    cmp_msgshift(ibufDev, obufDev, 0);
    nvtxRangePop();
    /*
     * obufDev is nxl by nrows by mp_p
     * copy into afftbufDev which is nxfull by nrows
     */
    nvtxRangePushA("1d_cufft_msg_rows_to");
    rows_to_omp45(obufDev, ibufDev, chunksize_x, nrows, mp_p);
    /* copy data to the array in unified memory */
    dense2dbl_omp45(ibufDev, tvarDblDev, nxfull, nrows, nzl);
    nvtxRangePop();

    nvtxRangePushA("1d_cufft_msg_cufft_x");
    /*.................... Batched 1d FFT in X-direction */
    if(isign == 1) {
        cufftExecZ2Z(plan_many_x_mpi, devDptr, devDptr, CUFFT_INVERSE);
    } else {
        cufftExecZ2Z(plan_many_x_mpi, devDptr, devDptr, CUFFT_FORWARD);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
    nvtxRangePushA("1d_cufft_msg_rows_from");
    /* copy data to the CPU array */
    dbl2dense_omp45(tvarDblDev, ibufDev, nxfull, nrows, nzl);

    /*
     * Copy from afftbufDev to obufDev. afftbufDev is nxfull by nrows
     * obufDev is nxl by nrows by mp_p
     */
    rows_from_omp45(ibufDev, obufDev, chunksize_x, nrows, mp_p);
    nvtxRangePop();
    /* 
     * The results of the FFT iare in
     * "alltoall order" as desired.
     */
    nvtxRangePushA("1d_cufft_msg_isend2");
    /* Use MPI_Isend to send half of all partial rows
       to the other MPI process in the same row of the decomp. */
    cmp_msgshift(obufDev, ibufDev, 0);
    nvtxRangePop();
    /* transpose back from obufDev to e */
    nvtxRangePushA("1d_cufft_msg_trans_from");
    dense2guard_omp45(ibufDev, e, nxl, nyl, nzl);
    nvtxRangePop();
}


void cu_fftw_premap(double complex * restrict tmp_dbcom)
{
  if(mp_rank == 0) printf("CU_FFTW: Copying tmp_dbcom to GPU\n");
#pragma omp target enter data map(to:tmp_dbcom[0:ngtot])
}

void cu_fftw_unmap(double complex * restrict tmp_dbcom)
{
  if(mp_rank == 0) printf("CU_FFTW: Copying tmp_dbcom back to CPU\n");
#pragma omp target exit data map(from:tmp_dbcom[0:ngtot])
}
