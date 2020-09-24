/* The rotth and 2D FFT functions have been extracted from pF3D.
   pf3dbench is used to test compiler optimization.
   pf3dbench makes use of C99 complex variables and
   OpenMP 4.5 target offload.
*/
#include <stdio.h>
#define __USE_UNIX98
#include <unistd.h>
#include <string.h>
#ifdef OMP45_BUILD
#include <cuda_profiler_api.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mytypes.h"
#include "lecuyer.h"
#include "util.h"
#include "time.h"

#include "light.h"

#include "pf3dbench.h"
#include "pf3d_fft.h"
#include "check.h"

#ifdef USE_MPI
#include "grid.h"
#endif

/* Declare storage for variables on the host */
#ifndef OMP45_BUILD
#define __parm_init__
#endif

#include "runparm.h"
#include "pf3dbenchvars.h"

void ProfilerStart(char *kernel);
void ProfilerStop(char *kernel);
void set_rank(void);

extern void var_check(char *vname, rcomplex *var);

extern void run_acppft(void);
extern void run_fft_x(void);
extern void run_fft(void);
extern void run_fft_noguard(void);
extern void run_fft_chek_x_loc(void);
extern void run_fft_chek_x_loc_fwd(void);
extern void run_fft_chek_x_loc_bkw(void);
extern void run_fft_chek_y_loc(void);
extern void run_fft_chek_loc(void);
extern void run_alltoall_chek(void);
extern void run_alltoall_chek_x(void);
extern void run_alltoall_chek_y(void);
extern void run_fft_chek_x(void);
extern void run_fft_chek_x_fwd(void);
extern void run_fft_chek_x_bkw(void);
extern void run_fft_chek_y(void);
extern void run_fft_2d_many(void);
extern void run_fft_1d_cufft_fwd(void);
extern void run_fft_1d_cufft(void);
extern void run_fft_2d_cufft(void);
extern void run_fft_2d_cufft_mpi(void);
extern void run_fft_1d_cufft_msg(void);
extern void run_copy2d(void);
extern void run_copy2d_a(void);
extern void run_copy2d_b(void);
extern void run_chek_dbl(void);
extern void run_chek_rows(void);
extern void run_chek_cols(void);
extern void run_chek_trans(void);
extern void run_move2d(void);
extern void run_move1d(void);
extern void run_trans2d(void);

extern void run_copy_xy(void);
extern void run_copy_x(void);
extern void run_copy_y(void);
extern void run_chek_dbl(void);
extern void run_chek_rows(void);
extern void run_chek_cols(void);
extern void run_chek_trans(void);
extern void run_move2d(void);
extern void run_move1d(void);

extern void run_fft_many_x(void);
extern void run_fft_many_x_fwd(void);
extern void run_fft_many_x_bkw(void);
extern void run_fft_many_mpi_x(void);
extern void run_fft_many_mpi_x_fwd(void);
extern void run_fft_many_mpi_x_bkw(void);

extern void run_rotth_z_merge(void);
extern void run_rotth_z_merge3(void);
extern void run_rotth_omp45(void);
extern void run_rotth_omp45_pre(void);
extern void run_rotth_omp45_pre3D(void);

extern void run_couple_z(void);
extern void run_couple_omp45_pre(void);

extern int nthr_kern;

#define thetb3D(a,b,c) thetb[CELTNDX3(a,b,c)]
#define t0_big(i,j,k) t0_big[(((k)*nyfull + (j))*nxfull + (i))]
#define tvar3D(a,b,c) tvar[CELTNDX3(a,b,c)]

double tot_tim;

double complex *tmp_dbcom= 0;

rcomplex *t0, *t2, *t0_sav, *t2_sav;
rcomplex *t0DevPtr= 0;
rcomplex *denlw= 0;

int nrows, ncols, nxfull, nyfull, nzfull;
int chunksize_x, chunksize_y;

real *iptmp=0, *optmp=0;
rcomplex *sndbuf=0, *rcvbuf=0;
/* buffer size is 2*nxl*nyl reals */
rcomplex *afftbuf= 0;

int chekndx[]= {0,0,0, 13,5,1, 117,20,17};
int nchek= 3;
int nzontot;
int do_csv= 1;
int make_ref= 0;
int read_ref= 1;
FILE *file_csv, *file_hdr, *file_ref;
char RUN_TYPE[100];

#define REAL_DATA 0
#define CMPLX_DATA 1

int max_threads(void)
{
  return  omp_maxthreads;
}

void ProfilerStart(char *kernel)
{
#ifdef USE_MPI
  simsync();
#endif
}

void ProfilerStop(char *kernel)
{
#ifdef USE_MPI
  simsync();
#endif
}

void run_test_cmplx_dev(void (*test) (void), char *kernel, int npass,
                 double *test_tim, double *test_wtim, char *varnam,
                 rcomplex *var, rcomplex *var_sav)
{
    double tim0, wtim0, timval, wtimval;
    int i;
  
    /* test with complex output of one array and no reset
       inside the iteration loop */
#ifdef OMP45_BUILD
    if(mp_rank == 0) {
      printf("\nabout to call %s\n", kernel);
      show_cmplx(varnam, var, chekndx, nchek);
      if(make_ref) {
        fprintf(file_ref, "%s\n", kernel);
        fprintf(file_ref, "%d\n", CMPLX_DATA);
        fprintf(file_ref, "%d\n", 1);
        fprintf(file_ref, "%d\n", nchek);
      }
    }
    /* Copy t0 to GPU memory */
    reset_tvar(t0DevPtr, var);
    rcmpPrefetch(t0DevPtr, ngtot);
    timval= wtimval= 0.0;
    ProfilerStart(kernel);
    for(i= 0; i < npass; i++) {
      tim0= second(0.0);
      wtim0= wsecond(0.0);
      test();
      wtimval += wsecond(0.0)-wtim0;
      timval += second(0.0)-tim0;
    }
    ProfilerStop(kernel);
    /* copy the variable back from device memory */
    reset_tvar(var, t0DevPtr);
    if(mp_rank == 0) {
      check_cmplx(kernel, varnam, var, var_sav, chekndx, nchek);
      printf("time for %s is %e\n", kernel, timval);
      printf("Walltime for %s is %e\n", kernel, wtimval);
      printf("%s:>> zones/sec/thread = %e, tot zones/sec = %e\n", kernel,
             npass*nzontot/wtimval/nthr_kern, npass*nzontot*mpi_cnt/wtimval);
      if(do_csv) {
        fprintf(file_csv, ", %e", npass*nzontot/wtimval/nthr_kern);
        fprintf(file_hdr, ", %s", kernel);
      }
    }
    *test_tim= timval;
    *test_wtim= wtimval;
#endif
}

void run_test_cmplx(void (*test) (void), char *kernel, int npass,
                 double *test_tim, double *test_wtim, char *varnam,
                 rcomplex *var, rcomplex *var_sav)
{
    double tim0, wtim0, timval, wtimval;
    int i;
  
    /* test with complex output of one array and no reset
       inside the iteration loop */
    if(mp_rank == 0) {
      printf("\nabout to call %s\n", kernel);
      show_cmplx(varnam, var, chekndx, nchek);
      if(make_ref) {
        fprintf(file_ref, "%s\n", kernel);
        fprintf(file_ref, "%d\n", CMPLX_DATA);
        fprintf(file_ref, "%d\n", 1);
        fprintf(file_ref, "%d\n", nchek);
      }
    }
    timval= wtimval= 0.0;
    ProfilerStart(kernel);
    for(i= 0; i < npass; i++) {
      tim0= second(0.0);
      wtim0= wsecond(0.0);
      test();
      wtimval += wsecond(0.0)-wtim0;
      timval += second(0.0)-tim0;
    }
    ProfilerStop(kernel);
    if(mp_rank == 0) {
      check_cmplx(kernel, varnam, var, var_sav, chekndx, nchek);
      printf("time for %s is %e\n", kernel, timval);
      printf("Walltime for %s is %e\n", kernel, wtimval);
      printf("%s:>> zones/sec/thread = %e, tot zones/sec = %e\n", kernel,
             npass*nzontot/wtimval/nthr_kern, npass*nzontot*mpi_cnt/wtimval);
      if(do_csv) {
        fprintf(file_csv, ", %e", npass*nzontot/wtimval/nthr_kern);
        fprintf(file_hdr, ", %s", kernel);
      }
    }
    *test_tim= timval;
    *test_wtim= wtimval;
}

void run_test_cmplx_dense(void (*test) (void), char *kernel, int npass,
                 double *test_tim, double *test_wtim, char *varnam,
                 rcomplex *var, rcomplex *var_sav)
{
    double tim0, wtim0, timval, wtimval;
    int i;
  
    /* Test with complex output of one array and no reset
       inside the iteration loop.
       NOTE - the input array has guard cells. Copy to a dense
       array, call the function, copy back to the input array,
       then check for correctness.
    */
    if(mp_rank == 0) {
      printf("\nabout to call %s\n", kernel);
      show_cmplx(varnam, var, chekndx, nchek);
      if(make_ref) {
        fprintf(file_ref, "%s\n", kernel);
        fprintf(file_ref, "%d\n", CMPLX_DATA);
        fprintf(file_ref, "%d\n", 1);
        fprintf(file_ref, "%d\n", nchek);
      }
    }
    timval= wtimval= 0.0;
    guard2dense(var, tN_new, nxl, nyl, nzl);
    ProfilerStart(kernel);
    for(i= 0; i < npass; i++) {
      tim0= second(0.0);
      wtim0= wsecond(0.0);
      test();
      wtimval += wsecond(0.0)-wtim0;
      timval += second(0.0)-tim0;
    }
    ProfilerStop(kernel);
    dense2guard(tN_new, var, nxl, nyl, nzl);
    if(mp_rank == 0) {
      check_cmplx(kernel, varnam, var, var_sav, chekndx, nchek);
      printf("time for %s is %e\n", kernel, timval);
      printf("Walltime for %s is %e\n", kernel, wtimval);
      printf("%s:>> zones/sec/thread = %e, tot zones/sec = %e\n", kernel,
             npass*nzontot/wtimval/nthr_kern, npass*nzontot*mpi_cnt/wtimval);
      if(do_csv) {
        fprintf(file_csv, ", %e", npass*nzontot/wtimval/nthr_kern);
        fprintf(file_hdr, ", %s", kernel);
      }
    }
    *test_tim= timval;
    *test_wtim= wtimval;
}

void run_test_couple(void (*test) (void), char *kernel, int npass,
                     double *test_tim, double *test_wtim, 
                     char *varnam1, rcomplex *var1, rcomplex *var1_sav,
                     char *varnam2, rcomplex *var2, rcomplex *var2_sav)
{
    double tim0, wtim0, timval, wtimval;
    int i;
  
    /* call light coupling */
    if(mp_rank == 0) {
      printf("\nabout to call %s\n", kernel);
      show_cmplx(varnam1, var1, chekndx, nchek);
      show_cmplx(varnam2, var2, chekndx, nchek);
      if(make_ref) {
        fprintf(file_ref, "%s\n", kernel);
        fprintf(file_ref, "%d\n", CMPLX_DATA);
        fprintf(file_ref, "%d\n", 2);
        fprintf(file_ref, "%d\n", nchek);
      }
    }
    timval= wtimval= 0.0;
    ProfilerStart(kernel);
    for(i= 0; i < npass; i++) {
      tim0= second(0.0);
      wtim0= wsecond(0.0);
      test();
      wtimval += wsecond(0.0)-wtim0;
      timval += second(0.0)-tim0;
    }
    ProfilerStop(kernel);
    if(mp_rank == 0) {
      check_cmplx(kernel, varnam1, var1, var1_sav, chekndx, nchek);
      check_cmplx(kernel, varnam2, var2, var2_sav, chekndx, nchek);
      printf("time for %s is %e\n", kernel, timval);
      printf("Walltime for %s is %e\n", kernel, wtimval);
      printf("%s:>> zones/sec/thread = %e, tot zones/sec = %e\n", kernel,
             npass*nzontot/wtimval/nthr_kern, npass*nzontot*mpi_cnt/wtimval);
      if(do_csv) {
        fprintf(file_csv, ", %e", npass*nzontot/wtimval/nthr_kern);
        fprintf(file_hdr, ", %s", kernel);
      }
    }
    *test_tim= timval;
    *test_wtim= wtimval;
}

int main(int argc, char **argv) {
  
  double tim2_0, tim0, wtim0;
  double trotth, wtrotth, trotth_omp45, wtrotth_omp45;
  double tcouple, wtcouple, tcpft, wtcpft;
  double wtim_start, wtim;
  char *kernel;

  long i, nrotth, ncouple, nacppft, nfft_loc, nfft_mpi, nfft_x_fwd;

  /* Start with all tests off */
  int do_rotth=0, do_rotth_omp45=0, do_rotth_omp45_pre3D= 0;
  int do_couple=0, do_couple_omp45=0, do_couple_omp45_pre3D= 0;
  int do_fft=0, do_fft_chek=0, do_fft_chek_xy=0;
  int do_fft_chek_x=0;
  int do_alltoall=0, do_copy_xy=0, do_move2d=0;
  int do_trans2d= 0, do_chek_rows= 0, do_chek_cols= 0, do_chek_trans=0;
  int do_fft_many_x=0, do_fft_2d_many=0, do_chek_dbl= 0;
  int do_fft_1d_cufft_fwd= 0, do_fft_1d_cufft= 0, do_fft_2d_cufft= 0;
  int do_fft_many_x_bkw= 0, do_fft_chek_x_bkw= 0;
  int do_fft_2d_mpi_cufft= 0, do_fft_1d_msg_cufft= 0;
#define MAXLEN 120
  char host[MAXLEN], fnam[MAXLEN];
  int errhost, maxfftlen;
  int idev;
  
  /* turn on the desired tests */
  do_fft= 1;
  do_fft_many_x= 1;
  do_fft_many_x_bkw= 0;
  do_fft_chek_x= 1;
  do_fft_chek_x_bkw= 0;
  do_fft_2d_many= 1;
  do_fft_1d_cufft_fwd= 0;
  do_fft_1d_cufft= 0;
  do_fft_2d_cufft= 0;
  do_fft_2d_mpi_cufft= 1;
  do_fft_1d_msg_cufft= 1;
  do_rotth= 1;
  do_rotth_omp45= 0;
  do_rotth_omp45_pre3D= 0;
  do_couple= 1;

#define SMALL_RUNS
#ifdef SMALL_RUNS
  /* SMALL_RUNS is used for debugging */
  nrotth= 2;
  ncouple= 2;
  nacppft= 2;
  nfft_loc= 2;
  nfft_x_fwd= 2;
  nfft_mpi= 2;
#else
  /* Larger loop counts are used for performance testing. */
  nrotth= 100;
  ncouple= 20;
  nacppft= 6;
  nfft_loc= 14;
  /* NOTE - a forward FFT is just a correctness test, so keep the
     iteration count down */
  nfft_x_fwd= 2;
  nfft_mpi= 5;
#endif

#ifdef USE_MPI
  /* ------------------------------ Initialize MPI */
  mp_p= 2;  mp_q= 2;  mp_r= 1;
  siminit( &argc, &argv );
  set_rank();    /* g3 now contains the rank and MPI size */
  mp_size= g3.nproc;
  mp_r = mp_size/mp_p/mp_q;
  if(!mp_rank) {
    if(mp_p*mp_q*mp_r != mp_size) {
      printf("ERROR: mp_size is %d but must be evenly divisible by %d\n", mp_size, mp_p*mp_q);
      exit(1);
    } else {
      printf("mp_p, mp_q, and mp_r are consistent with the number of processes\n");
    }
  }
  build_grid();
  simsync();   /* do not call simsync until after build_grid */
#else
  /* non-MPI version */
  set_rank();
#endif
#ifdef OMP45_BUILD
  /* idev= omp_get_initial_device(); */
  idev= omp_get_default_device();
  /* idev= omp_get_device_num(); */
  printf("MPI rank %d is using GPU device %d\n", mp_rank, idev);
#else
  if(mp_rank == 0) printf("Number of MPI ranks is %d\n", mp_size);
#endif
  
  parse_args(argc, argv);

  parm_init();
#ifdef USE_MPI
  fflush(stdout);
  simsync();
#endif
  if(mp_rank == 0) printf("\nDomain size is (%d,%d,%d)\n", nxl, nyl, nzl);
  if(mp_rank == 0) printf("\nDecomposition is (%d,%d,%d)\n", mp_p, mp_q, mp_r);

  /* allocate arrays including acw and lw */
  tim2_0= wsecond(0.0);
  nbig= nxl*mp_p*nyl*mp_q*nzl;
  do_init(nxl, nyl, nzl, num_thr);
  tim2_0= wsecond(0.0)-tim2_0;
  if(mp_rank == 0) printf("initialization time is %e\n", tim2_0);
  nzontot= nxl*nyl*nzl;
  /* retrieve the hostname */
  errhost= gethostname(host, MAXLEN);

  nrows= nyl/g3.P;
  ncols= nxl/g3.Q;
  nxfull= nxl*mp_p;
  nyfull= nyl*mp_q;
  chunksize_x= nxl*nrows;
  chunksize_y= ncols*nyl;

  /* make temp vectors for FFTW */
  maxfftlen= nxfull;
  if(nyfull > maxfftlen) maxfftlen= nyfull;
  
  /* read reference values if so requested */
  if(!mp_rank && read_ref) {
    getref(150);
  }

  /* Only MPI rank zero should write a csv file */
  if(mp_rank == 0) do_csv= 1;
  else do_csv= 0;
  /* Only MPI rank zero should write a reference file */
  make_ref= 0;
#ifdef BUILD_REF
  if(mp_rank == 0) make_ref= 1;
#endif
  simsync();  /* rank zero might fall behind due to reading ref file */
  
  if(do_csv) {
    /* starting and stopping the omp timer will set the number of
       OMP threads per process */
    start_omp_time();
    stop_omp_time();
#ifdef USE_DOUBLE
    sprintf(fnam, "results_dbl_%s%s.csv", host, RUN_TYPE);
#else
    sprintf(fnam, "results_%s%s.csv", host, RUN_TYPE);
#endif
    file_csv= fopen(fnam, "w");
#ifdef USE_DOUBLE
    sprintf(fnam, "resnames_dbl_%s%s.csv", host, RUN_TYPE);
#else
    sprintf(fnam, "resnames_%s%s.csv", host, RUN_TYPE);
#endif
    file_hdr= fopen(fnam, "w");
#ifdef USE_DOUBLE
    fprintf(file_csv, "%s, %s, %d, %d, %d, %d, %d, %d, %d, %d ", host, "double", mp_size, nthr_kern, mp_p, mp_q, mp_r, nxl, nyl, nzl);
#else
    fprintf(file_csv, "%s, %s, %d, %d, %d, %d, %d, %d, %d, %d ", host, "float", mp_size, nthr_kern, mp_p, mp_q, mp_r, nxl, nyl, nzl);
#endif
    fprintf(file_hdr, "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s ", "hostname", "precision", "mp_size", "nthr_kern", "mp_p", "mp_q", "mp_r", "nxl", "nyl", "nzl");
  }

  if(make_ref) {
#ifdef OMP45_BUILD
    sprintf(fnam, "reference_vals_omp45.txt");
#else
    sprintf(fnam, "reference_vals.txt");
#endif
    file_ref= fopen(fnam, "w");
  }
  
  /* CPU time */
  tim0= second(0.0);
  /* setting up start time */
  tim2_0= wsecond(0.0);
  wtim_start= tim2_0;
  if(mp_rank == 0) printf("time before first kernel is %e\n", tim0);

  /* At this point everything is almost ready to use.
     The final (optional) step is to override the values in t0
     with values from an nxfuull-by-nyfull-by-nzl array and
     thereby be able to check multi-domain 2D FFTs (you have to
     know values from other domains to check them on MPI
     rank zero). */
#define GLOBAL_t0
#ifdef GLOBAL_t0
  t0_fixup();
#endif

  /* Load "read-only" values onto the GPU, SCALARS first. */
  load_scalars();
  load_arrays();
  
  
  fflush(stdout);
  if(do_couple) {
    kernel= "couple_z";
    /* call complex wave coupling routine */
    reset_tvar(t0, t0_sav);
    reset_tvar(t2, t2_sav);
    run_test_couple(run_couple_z, kernel, ncouple,
                   &tcouple , &wtcouple, "t0", t0, t0_sav, "t2", t2, t2_sav);
  }

  if(do_couple_omp45) {
#ifdef OMP45_BUILD
    kernel= "couple_waves_omp45_pre";
    /* call 3 wave coupling routine */
    reset_light();
    if(mp_rank == 0) {
      printf("\nabout to call %s\n", kernel);
      show_cmplx("t0", t0, chekndx, nchek);
      show_cmplx("t2", t2, chekndx, nchek);
      if(make_ref) {
        fprintf(file_ref, "%s\n", kernel);
        fprintf(file_ref, "%d\n", CMPLX_DATA);
        fprintf(file_ref, "%d\n", 2);
        fprintf(file_ref, "%d\n", nchek);
      }
    }
    tcouple_omp45= wtcouple_omp45= 0.0;
    couplewaves__premap(t0, t2, denp);
    ProfilerStart(kernel);
    for(i= 0; i < ncouple; i++) {
      tim0= second(0.0);
      wtim0= wsecond(0.0);
      run_couple_omp45_pre();
      wtcouple_omp45 += wsecond(0.0)-wtim0;
      tcouple_omp45 += second(0.0)-tim0;
    }
    ProfilerStop(kernel);
    couple_waves_unmap(t0, t2, denp);
    if(mp_rank == 0) {
      check_cmplx(kernel, "t0", t0, t0_sav, chekndx, nchek);
      check_cmplx(kernel, "t2", t2, t2_sav, chekndx, nchek);
      printf("time for %s is %e\n", kernel, tcouple);
      printf("wall time for %s is %e\n", kernel, wtcouple);
      printf("%s:>> zones/sec/thread = %e, tot zones/sec = %e\n", kernel, ncouple*nzontot/wtcouple/nthr_kern, ncouple*nzontot*mpi_cnt/wtcouple);
      if(do_csv) {
        fprintf(file_csv, ", %e", ncouple*nzontot/wtcouple/nthr_kern);
        fprintf(file_hdr, ", %s", kernel);
      }
    }
#endif
  }

  
  if(do_rotth) {
    kernel= "rotth_z_merge";
    /* call complex angle rotation routine */
    reset_tvar(t0, t0_sav);
    run_test_cmplx_dense(run_rotth_z_merge, kernel, nrotth,
                   &trotth , &wtrotth, "t0", t0, t0_sav);
  }
  
  if(do_rotth) {
    kernel= "rotth_z_merge3";
    /* call complex angle rotation routine */
    reset_tvar(t0, t0_sav);
    run_test_cmplx_dense(run_rotth_z_merge3, kernel, nrotth,
                   &trotth , &wtrotth, "t0", t0, t0_sav);
  }
  
  if(do_rotth) {
#ifdef OMP45_BUILD
    kernel= "rotth_omp45";
    /* call complex angle rotation routine */
    reset_tvar(t0, t0_sav);
    if(mp_rank == 0) {
      printf("\nabout to call %s\n", kernel);
      show_cmplx("t0", t0, chekndx, nchek);
      if(make_ref) {
        fprintf(file_ref, "%s\n", kernel);
        fprintf(file_ref, "%d\n", CMPLX_DATA);
        fprintf(file_ref, "%d\n", 1);
        fprintf(file_ref, "%d\n", nchek);
      }
    }
    trotth_omp45= wtrotth_omp45= 0.0;
    copy_to_tN(t0, tN_new);
    ProfilerStart(kernel);
    for(i= 0; i < nrotth; i++) {
      tim0= second(0.0);
      wtim0= wsecond(0.0);
      run_rotth_omp45();
      wtrotth_omp45 += wsecond(0.0)-wtim0;
      trotth_omp45 += second(0.0)-tim0;
    }
    ProfilerStop(kernel);
    copy_from_tN(tN_new, t0);
    if(mp_rank == 0) {
      check_cmplx(kernel, "t0", t0, t0_sav, chekndx, nchek);
      printf("time for %s is %e\n", kernel, trotth);
      printf("wall time for %s is %e\n", kernel, wtrotth);
      printf("%s:>> zones/sec/thread = %e, tot zones/sec = %e\n", kernel, nrotth*nxl*nyl/wtrotth/nthr_kern, nrotth*nxl*nyl*mpi_cnt/wtrotth);
      if(do_csv) {
        fprintf(file_csv, ", %e", nrotth*nzontot/wtrotth/nthr_kern);
        fprintf(file_hdr, ", %s", kernel);
      }
    }
#endif
  }
  
  if(do_rotth_omp45) {
#ifdef OMP45_BUILD
    kernel= "rotth_omp45_pre";
    /* call complex angle rotation routine */
    reset_tvar(t0, t0_sav);
    if(mp_rank == 0) {
      printf("\nabout to call %s\n", kernel);
      show_cmplx("t0", t0, chekndx, nchek);
      if(make_ref) {
        fprintf(file_ref, "%s\n", kernel);
        fprintf(file_ref, "%d\n", CMPLX_DATA);
        fprintf(file_ref, "%d\n", 1);
        fprintf(file_ref, "%d\n", nchek);
      }
    }
    trotth_omp45= wtrotth_omp45= 0.0;
    copy_to_tN(t0, tN_new);
    rotth_premap(tN_new, thetb);
    ProfilerStart(kernel);
    for(i= 0; i < nrotth; i++) {
      tim0= second(0.0);
      wtim0= wsecond(0.0);
      run_rotth_omp45_pre();
      wtrotth_omp45 += wsecond(0.0)-wtim0;
      trotth_omp45 += second(0.0)-tim0;
    }
    ProfilerStop(kernel);
    rotth_unmap(tN_new, thetb);
    copy_from_tN(tN_new, t0);
    if(mp_rank == 0) {
      check_cmplx(kernel, "t0", t0, t0_sav, chekndx, nchek);
      printf("time for %s is %e\n", kernel, trotth);
      printf("wall time for %s is %e\n", kernel, wtrotth);
      printf("%s:>> zones/sec/thread = %e, tot zones/sec = %e\n", kernel, nrotth*nxl*nyl/wtrotth/nthr_kern, nrotth*nxl*nyl*mpi_cnt/wtrotth);
      if(do_csv) {
        fprintf(file_csv, ", %e", nrotth*nzontot/wtrotth/nthr_kern);
        fprintf(file_hdr, ", %s", kernel);
      }
    }
#endif
  }
  
  if(do_rotth_omp45_pre3D) {
#ifdef OMP45_BUILD
    kernel= "rotth_omp45_pre3D";
    /* call complex angle rotation routine */
    reset_tvar(t0, t0_sav);
    if(mp_rank == 0) {
      printf("\nabout to call %s\n", kernel);
      show_cmplx("t0", t0, chekndx, nchek);
      if(make_ref) {
        fprintf(file_ref, "%s\n", kernel);
        fprintf(file_ref, "%d\n", CMPLX_DATA);
        fprintf(file_ref, "%d\n", 1);
        fprintf(file_ref, "%d\n", nchek);
      }
    }
    trotth_omp45= wtrotth_omp45= 0.0;
    guard2dense(t0, tN_new, nxl, nyl, nzl);
    rotth_premap(tN_new, thetb);
    ProfilerStart(kernel);
    for(i= 0; i < nrotth; i++) {
      tim0= second(0.0);
      wtim0= wsecond(0.0);
      run_rotth_omp45_pre3D();
      wtrotth_omp45 += wsecond(0.0)-wtim0;
      trotth_omp45 += second(0.0)-tim0;
    }
    ProfilerStop(kernel);
    rotth_unmap(tN_new, thetb);
    dense2guard(tN_new, t0, nxl, nyl, nzl);
    if(mp_rank == 0) {
      check_cmplx(kernel, "t0", t0, t0_sav, chekndx, nchek);
      printf("time for %s is %e\n", kernel, trotth);
      printf("wall time for %s is %e\n", kernel, wtrotth);
      printf("%s:>> zones/sec/thread = %e, tot zones/sec = %e\n", kernel, nrotth*nxl*nyl/wtrotth/nthr_kern, nrotth*nxl*nyl*mpi_cnt/wtrotth);
      if(do_csv) {
        fprintf(file_csv, ", %e", nrotth*nzontot/wtrotth/nthr_kern);
        fprintf(file_hdr, ", %s", kernel);
      }
    }
#endif
  }


  fflush(stdout);
  if(do_copy_xy) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_copy_xy";
    reset_light();    
    run_test_cmplx(run_copy_xy, kernel, 1,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }

  if(do_chek_dbl) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_chek_dbl";
    reset_light();    
    run_test_cmplx(run_chek_dbl, kernel, 1,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }

  if(do_chek_rows) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_chek_rows";
    reset_light();    
    run_test_cmplx(run_chek_rows, kernel, 1,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }

  if(do_chek_cols) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_chek_cols";
    reset_light();    
    run_test_cmplx(run_chek_cols, kernel, 1,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }

  if(do_chek_trans) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_chek_tran";
    reset_light();    
    run_test_cmplx(run_chek_trans, kernel, 1,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }

  if(do_move2d) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_move2d";
    reset_light();    
    run_test_cmplx(run_move2d, kernel, 1,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }


  fflush(stdout);

if(do_fft_many_x) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_fft_many_x_fwd";
    reset_light();    
    run_test_cmplx(run_fft_many_x_fwd, kernel, nfft_x_fwd,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }

if(do_fft_many_x_bkw) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_fft_many_x_bkw";
    reset_light();    
    run_test_cmplx(run_fft_many_x_bkw, kernel, nfft_x_fwd,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }

#if 0
if(do_fft_many_x) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_fft_many_x";
    reset_light();    
    run_test_cmplx(run_fft_many_x, kernel, nfft_loc,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }
#endif
  
  if(do_fft_1d_cufft_fwd) {
#ifdef OMP45_BUILD
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_fft_1d_cufft_fwd";
    reset_light();
    /* Copy t0 to t0DevPtr on the host, then sync to the device. */
    copy_carr3d(t0, t0DevPtr);
    rcmpPrefetch(t0DevPtr, ngtot);
    /* Turn on nvprof profiling for this funciton. */
    cudaProfilerStart();
    run_test_cmplx_dev(run_fft_1d_cufft_fwd, kernel, nfft_x_fwd,
                       &tcpft , &wtcpft, "t0", t0, t0_sav);
    cudaProfilerStop();
#endif
  }
  
  if(do_fft_1d_cufft) {
#ifdef OMP45_BUILD
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_fft_1d_cufft";
    reset_light();    
    /* Copy t0 to t0DevPtr on the host, then sync to the device. */
    copy_carr3d(t0, t0DevPtr);
    rcmpPrefetch(t0DevPtr, ngtot);
    /* Turn on nvprof profiling for this funciton. */
    cudaProfilerStart();
    run_test_cmplx_dev(run_fft_1d_cufft, kernel, nfft_x_fwd,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
   cudaProfilerStop();
#endif
  }
  
  if(do_fft_2d_cufft) {
#ifdef OMP45_BUILD
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_fft_2d_cufft";
    reset_light();    
    /* Copy t0 to t0DevPtr on the host, then sync to the device. */
    copy_carr3d(t0, t0DevPtr);
    rcmpPrefetch(t0DevPtr, ngtot);
    /* Turn on nvprof profiling for this funciton. */
    cudaProfilerStart();
    run_test_cmplx_dev(run_fft_2d_cufft, kernel, nfft_mpi,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
   cudaProfilerStop();
#endif
  }
  
  fflush(stdout);
  if(do_fft_chek_x) {
    kernel= "fft_chek_x_fwd";
    /* Must create correct private temporaries before the FFT can be run
       on multiple threads. In particular, each thread needs to have
       its own fftw_plan. */
    /* call complex FFT wrapper */
    reset_light();
    run_test_cmplx(run_fft_chek_x_fwd, kernel, nfft_x_fwd,
                 &tcpft , &wtcpft, "t0", t0, t0_sav);
  }

  if(do_fft_many_x) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_fft_many_mpi_x_fwd";
    reset_light();    
    run_test_cmplx(run_fft_many_mpi_x_fwd, kernel, nfft_x_fwd,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }
  
  if(do_fft_chek_x_bkw) {
    kernel= "fft_chek_x_bkw";
    /* Must create correct private temporaries before the FFT can be run
       on multiple threads. In particular, each thread needs to have
       its own fftw_plan. */
    /* call complex FFT wrapper */
    reset_light();
    run_test_cmplx(run_fft_chek_x_bkw, kernel, nfft_x_fwd,
                 &tcpft , &wtcpft, "t0", t0, t0_sav);
  }

  if(do_fft_many_x_bkw) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_fft_many_mpi_x_bkw";
    reset_light();    
    run_test_cmplx(run_fft_many_mpi_x_bkw, kernel, nfft_x_fwd,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }

#if 0
if(do_fft_many_x) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_fft_many_mpi_x";
    reset_light();    
    run_test_cmplx(run_fft_many_mpi_x, kernel, nfft_mpi,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }
#endif
  
  if(do_fft_2d_many) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_fft_2d_many";
    reset_light();    
    run_test_cmplx(run_fft_2d_many, kernel, nfft_mpi,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
  }
  
  if(do_fft_chek) {
    kernel= "fft_chek_y_loc";
    /* Must create correct private temporaries before the FFT can be run
       on multiple threads. In particular, each thread needs to have
       its own fftw_plan. */
    /* call complex FFT wrapper */
    reset_light();
    run_test_cmplx(run_fft_chek_y_loc, kernel, nacppft,
                 &tcpft , &wtcpft, "t0", t0, t0_sav);
  }
  
  if(do_fft_2d_mpi_cufft) {
#ifdef OMP45_BUILD
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_fft_2d_mpi_cufft";
    reset_light();    
    /* Copy t0 to t0DevPtr on the host, then sync to the device. */
    copy_carr3d(t0, t0DevPtr);
    rcmpPrefetch(t0DevPtr, ngtot);
    /* Turn on nvprof profiling for this funciton. */
    cudaProfilerStart();
    run_test_cmplx_dev(run_fft_2d_cufft_mpi, kernel, nfft_mpi,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
    cudaProfilerStop();
#endif
  }
  
  if(do_fft_1d_msg_cufft) {
#ifdef OMP45_BUILD
#ifdef USE_MPI
    simsync();
#endif
    kernel= "test_fft_1d_msg_cufft";
    reset_light();    
    /* Copy t0 to t0DevPtr on the host, then sync to the device. */
    copy_carr3d(t0, t0DevPtr);
    rcmpPrefetch(t0DevPtr, ngtot);
    /* Turn on nvprof profiling for this funciton. */
    cudaProfilerStart();
    run_test_cmplx_dev(run_fft_1d_cufft_msg, kernel, nfft_mpi,
                   &tcpft , &wtcpft, "t0", t0, t0_sav);
    cudaProfilerStop();
#endif
  }

  if(do_fft_chek) {
    kernel= "fft_chek_loc";
    /* call complex FFT wrapper */
    reset_light();
    run_test_cmplx(run_fft_chek_loc, kernel, nacppft,
                 &tcpft , &wtcpft, "t0", t0, t0_sav);
  }

  
  if(do_alltoall) {
#ifdef USE_MPI
    simsync();
#endif
    kernel= "alltoall_chek";
    reset_light();
    prep_alltoall(t0, nxl, nyl, nzl);
    run_test_cmplx(run_alltoall_chek, kernel, 1,
                   &tcpft, &wtcpft, "t0", t0, t0_sav);
  }

  
#if 0
  if(do_fft_chek_x) {
    kernel= "fft_chek_x";
    /* Must create correct private temporaries before the FFT can be run
       on multiple threads. In particular, each thread needs to have
       its own fftw_plan. */
    /* call complex FFT wrapper */
    reset_light();
    run_test_cmplx(run_fft_chek_x, kernel, nacppft,
                 &tcpft, &wtcpft, "t0", t0, t0_sav);
  }
#endif
  
  if(do_fft_chek) {
    kernel= "fft_chek_y";
    /* Must create correct private temporaries before the FFT can be run
       on multiple threads. In particular, each thread needs to have
       its own fftw_plan. */
    /* call complex FFT wrapper */
    reset_light();
    run_test_cmplx(run_fft_chek_y, kernel, nacppft,
                 &tcpft, &wtcpft, "t0", t0, t0_sav);
  }

  if(do_fft) {
    rcomplex tst1, tst2, tst3;
    real rat1, rat2, rat3;
    
    kernel= "fft1D_x";
    /* Must create correct private temporaries before the FFT can be run
       on multiple threads. In particular, each thread needs to have
       its own fftw_plan. */
    /* call complex FFT wrapper */
    reset_light();
    run_test_cmplx(run_fft_x, kernel, nacppft,
                 &tcpft, &wtcpft, "t0", t0, t0_sav);
  }

  if(do_fft) {
    rcomplex tst1, tst2, tst3;
    real rat1, rat2, rat3;
    
    kernel= "fft2D";
    /* Must create correct private temporaries before the FFT can be run
       on multiple threads. In particular, each thread needs to have
       its own fftw_plan. */
    /* call complex FFT wrapper */
    reset_light();
    run_test_cmplx(run_fft, kernel, nacppft,
                 &tcpft, &wtcpft, "t0", t0, t0_sav);
  }

  if(do_fft) {
    rcomplex tst1, tst2, tst3;
    real rat1, rat2, rat3;
    
    kernel= "fft2D_noguard";
    /* Must create correct private temporaries before the FFT can be run
       on multiple threads. In particular, each thread needs to have
       its own fftw_plan. */
    /* call complex FFT wrapper */
    reset_light();
    run_test_cmplx(run_fft_noguard, kernel, nacppft,
                 &tcpft, &wtcpft, "t0", t0, t0_sav);
  }


  if(do_csv) {
    fprintf(file_csv, " \n");
    fprintf(file_hdr, " \n");
    fclose(file_csv);
    fclose(file_hdr);
    if(make_ref) fclose(file_ref);
  }
   
#ifdef USE_MPI
  fflush(stdout);
  simsync();
#endif
  if(mp_rank == 0) printf("\ntime after all kernels is %e\n\n\n\n", second(0.0)-tim0);
  
  /* free allocated storage */
  if(0) {
    do_cleanup();
  }

  return 0;
}

void reset_light(void)
{
    /* reset light variables to their initial state */
    copy_wave(t0, t0_sav);
    copy_wave(t2, t2_sav);
}

void reset_t0_big(rcomplex *tbig, rcomplex *tbig_sav)
{
    long ii;
    
    /* reset a light variable to its initial state */
    for(ii= 0; ii < nbig; ii++) {
      tbig[ii]= tbig_sav[ii];
    }
}

void reset_tvar(rcomplex *tvar, rcomplex *tvar_sav)
{
    long ii;
    
    /* reset a light variable to its initial state */
    for(ii= 0; ii < ngtot; ii++) {
      tvar[ii]= tvar_sav[ii];
    }
}

void run_copy_xy(void)
{
    copy_xy(t0);
}

void run_chek_dbl(void)
{
    chek_dbl(t0);
}

void run_chek_rows(void)
{
    chek_rows(t0);
}

void run_chek_cols(void)
{
    chek_cols(t0);
}

void run_chek_trans(void)
{
    chek_trans(t0);
}

void run_move2d(void)
{
    move_2D(t0);
}

void run_fft_1d_cufft_fwd(void)
{
    rcomplex *el;

#ifdef OMP45_BUILD
    el= t0DevPtr;
    /* batched FFT in the x-direction */
    fft_1d_cufft(el, +1, nxl, nyl, nzl);
#endif
}

void run_fft_1d_cufft(void)
{
    rcomplex *el;

#ifdef OMP45_BUILD
    el= t0DevPtr;
    /* batched FFT in the x-direction */
    fft_1d_cufft(el, +1, nxl, nyl, nzl);
    fft_1d_cufft(el, -1, nxl, nyl, nzl);
    /* FFT to and from k-mode space complete. Normalize */
    fft_norm_omp45(el, nxl, nyl, nzl, nxl, 1);
#endif
}

void run_fft_2d_cufft(void)
{
    rcomplex *el;

#ifdef OMP45_BUILD
    el= t0DevPtr;
    fft_2d_cufft(el, nxfull, nyfull, +1, nxl, nyl, nzl);
    fft_2d_cufft(el, nxfull, nyfull, -1, nxl, nyl, nzl);
    /* FFT to and from k-mode space complete. Normalize */
    fft_norm_omp45(el, nxl, nyl, nzl, nxl, nyl);
#endif
}

void run_fft_2d_cufft_mpi(void)
{
    rcomplex *el;

#ifdef OMP45_BUILD
    el= t0DevPtr;
    fft_2d_cufft_mpi(el, nxfull, nyfull, +1, nxl, nyl, nzl);
    fft_2d_cufft_mpi(el, nxfull, nyfull, -1, nxl, nyl, nzl);
    /* FFT to and from k-mode space complete. Normalize */
    fft_norm_omp45(el, nxl, nyl, nzl, nxl, nyl);
#endif
}

void run_fft_1d_cufft_msg(void)
{
    rcomplex *el;

#ifdef OMP45_BUILD
    el= t0DevPtr;
    fft_1d_cufft_msg(el, nxfull, +1, nxl, nyl, nzl);
    fft_1d_cufft_msg(el, nxfull, -1, nxl, nyl, nzl);
    /* FFT to and from k-mode space complete. Normalize */
    fft_norm_omp45(el, nxl, nyl, nzl, nxl, nyl);
#endif
}

void run_fft_2d_many(void)
{
    FFT_2d_many(t0, +1);
    FFT_2d_many(t0, -1);
    /* FFT to and from k-mode space complete. Normalize */
    fft_norm(t0, nxl, nyl, nzl, nxl, nyl);
}

void run_fft_many_x_fwd(void)
{
    fft_1D_batch_x(t0, nxl, nyl, +1, nzl);
}

void run_fft_many_x_bkw(void)
{
    fft_1D_batch_x(t0, nxl, nyl, -1, nzl);
}

void run_fft_many_x(void)
{
    rcomplex *el;

    el= t0_big;
    fft_1D_batch_x(el, nxl, nyl, +1, nzl);
    fft_1D_batch_x(el, nxl, nyl, -1, nzl);
    /* FFT to and from k-mode space complete. Normalize */
    fft_norm(el, nxl, nyl, nzl, nxl, 1); /* fft in x direction only */
}

void run_fft_many_mpi_x(void)
{
    fft_1D_batch_mpi_x(t0, nxfull, nrows, +1, nxl, nyl, nzl);
    fft_1D_batch_mpi_x(t0, nxfull, nrows, -1, nxl, nyl, nzl);
    /* FFT to and from k-mode space complete. Normalize */
    fft_norm(t0, nxl, nyl, nzl, nxfull, 1); /* fft in x direction only */
}

void run_fft_many_mpi_x_fwd(void)
{
    fft_1D_batch_mpi_x(t0, nxfull, nrows, +1, nxl, nyl, nzl);
}

void run_fft_many_mpi_x_bkw(void)
{
    fft_1D_batch_mpi_x(t0, nxfull, nrows, -1, nxl, nyl, nzl);
}

void run_fft_chek_x_loc(void)
{
    FFT_chek_x_loc(t0, +1);
    FFT_chek_x_loc(t0, -1);
    /* FFT to and from k-mode space complete. Normalize */
    fft_norm(t0, nxl, nyl, nzl, nxl, 1); /* fft in x direction only */
}

void run_fft_chek_x_loc_fwd(void)
{
    FFT_chek_x_loc(t0, +1);
}

void run_fft_chek_x_loc_bkw(void)
{
    FFT_chek_x_loc(t0, -1);
}

void run_fft_chek_y_loc(void)
{
    FFT_chek_y_loc(t0, +1);
    FFT_chek_y_loc(t0, -1);
    /* FFT to and from k-mode space complete. Normalize */
    fft_norm(t0, nxl, nyl, nzl, nyl, 1); /* fft in y direction only */
}

void run_fft_chek_loc(void)
{
    FFT_chek_loc(t0, +1);
    FFT_chek_loc(t0, -1);
    /* FFT to and from k-mode space complete. Normalize */
    fft_norm(t0, nxl, nyl, nzl, nxl, nyl);
}

void run_alltoall_chek(void)
{
    alltoall_chek(t0);
}

void run_fft_chek_x_fwd(void)
{
    FFT_chek_x(t0, +1);
}

void run_fft_chek_x_bkw(void)
{
    FFT_chek_x(t0, -1);
    /* FFT to and from k-mode space complete. Normalize */
}

void run_fft_chek_x(void)
{
    FFT_chek_x(t0, +1);
    FFT_chek_x(t0, -1);
    /* FFT to and from k-mode space complete. Normalize */
#ifdef USE_MPI
    fft_norm(t0, nxl, nyl, nzl, nxfull, 1); /* fft in x direction only */
#else
    fft_norm(t0, nxl, nyl, nzl, nxl, 1); /* fft in x direction only */
#endif
}

void run_fft_x(void)
{
    FFT1D_x(t0, +1);
    FFT1D_x(t0, -1);
    /* FFT to and from k-mode space complete. Normalize */
#ifdef USE_MPI
    fft_norm(t0, nxl, nyl, nzl, nxfull, nyfull);
#else
    fft_norm(t0, nxl, nyl, nzl, nxl, nyl);
#endif
}

void run_fft(void)
{
    FFT2D(t0, +1);
    FFT2D(t0, -1);
    /* FFT to and from k-mode space complete. Normalize */
#ifdef USE_MPI
    fft_norm(t0, nxl, nyl, nzl, nxfull, nyfull);
#else
    fft_norm(t0, nxl, nyl, nzl, nxl, nyl);
#endif
}

void run_fft_noguard(void)
{
    guard2dense(t0, tN_new, nxl, nyl, nzl);
    FFT2D_noguard(tN_new, +1);
    FFT2D_noguard(tN_new, -1);
    dense2guard(tN_new, t0, nxl, nyl, nzl);
    /* FFT to and from k-mode space complete. Normalize */
#ifdef USE_MPI
    fft_norm(t0, nxl, nyl, nzl, nxfull, nyfull);
#else
    fft_norm(t0, nxl, nyl, nzl, nxl, nyl);
#endif
}

void run_fft_chek_y(void)
{
    FFT_chek_y(t0, +1);
    FFT_chek_y(t0, -1);
    /* FFT to and from k-mode space complete. Normalize */
#ifdef USE_MPI
    fft_norm(t0, nxl, nyl, nzl, 1, nyfull); /* fft in y direction only */
#else
    fft_norm(t0, nxl, nyl, nzl, 1, nyl); /* fft in y direction only */
#endif
}


void run_couple_z(void)
{
    /* call complex wave coupling routine */
    couple_z(t0, t2, denlw);
}

void run_couple_omp45_pre(void)
{
    /* call complex wave coupling routine */
#ifdef OMP45_BUILD
    couple_omp45_pre(t0, t2, denlw);
#endif
}


void run_rotth_z_merge(void)
{
    /* call complex field rotation routine */
    rotth_z_merge(tN_new, thetb, 0, nzl-1);
}

void run_rotth_z_merge3(void)
{
    /* call complex field rotation routine */
    rotth_z_merge3(tN_new, thetb, 0, nzl-1);
}

void run_rotth_omp45(void)
{
    /* call complex field rotation routine */
#ifdef OMP45_BUILD
    rotth_mult_omp45(nzl, tN_new, thetb);
#endif
}

void run_rotth_omp45_pre(void)
{
    /* call complex field rotation routine */
#ifdef OMP45_BUILD
    rotth_mult_omp45_pre(nzl, tN_new, thetb);
#endif
}

void run_rotth_omp45_pre3D(void)
{
    /* call complex field rotation routine */
#ifdef OMP45_BUILD
    rotth_omp45_pre3D(nzl, tN_new, thetb, 0, nzl-1);
#endif
}


#ifdef USE_MPI
void set_rank(void)
{
  mp_rank= g3.me;
  mp_size= g3.nproc;
}
#else
void set_rank(void)
{
  mp_rank= 0;
  mp_size= 1;
}
#endif

void t0_fixup(void)
{
  int i, j, k, ibase, jbase;
  double t0_lo, t0_hi;
  
  /* This function is called after other initialization is complete.
     It allocates and initializes an nxfull-by-nyfull-by-nzl
     complex array with values suitable for t0.
     It then copies the portion of this "full array" that belongs
     to this MPI domain into t0.
     Note that t0 has guard zones and t0_big does NOT.
  */
  t0_lo=   0.25;
  t0_hi=   0.75;
  t0_big= init_complex(nbig, t0_lo, t0_hi);
  /* set the starting point in the big array for the zones
     "owned" by this process for both t0 and t0_sav. */
  ibase= nxl*mp_myp;
  jbase= nyl*mp_myq;
  
  for(k= 0; k < nzl; k++) {
    for(j= 0; j < nyl; j++) {
      for(i= 0; i < nxl; i++) {
        t0(i,j,k)= t0_big(i+ibase, j+jbase, k);
        t0_sav(i,j,k)= t0_big(i+ibase, j+jbase, k);
      }
    }
  }
  /* preserve the initial values of t0_big */
  t0_big_sav= (rcomplex *)make_real(2*nbig);
  for(i= 0; i < nbig; i++) {
    t0_big_sav[i]= t0_big[i];
  }
}

void big_quad_to_tvar(rcomplex *tbig, rcomplex *tvar)
{
  int i, j, k, ibase, jbase;
  double t0_lo, t0_hi;
  
  /* This function takes an nxfull-by-nyfull-by-nzl array
     and copies the quadrant "belonging" to this MPI process
     into tvar.
     Note that NEITHER tvar or t0_big has guard zones.
  */
  /* set the starting point in the big array for the zones
     "owned" by this process. */
  ibase= nxl*mp_myp;
  jbase= nyl*mp_myq;
  
  for(k= 0; k < nzl; k++) {
    for(j= 0; j < nyl; j++) {
      for(i= 0; i < nxl; i++) {
        tvar3D(i,j,k)= t0_big(i+ibase, j+jbase, k);
      }
    }
  }
}
