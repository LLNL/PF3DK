/* This file contains definitions for memory initialization and
   allocation and other helper functions.
*/

extern double *dtmp;
extern size_t totalloc;
extern long errcnt;
extern double tpi;

#define wmalloc base_malloc

extern void init_random(int flag);
extern void parse_args(int argc, char **argv);
extern void *wmalloc(size_t nbytes);
extern int *init_int(long n, int lo, int hi);
extern real *make_real(long n);
extern real *init_real(long n, double lo, double hi);
extern real *init_real_ramp(int numx, int numy, int numz,
                            double lo, double hi);
extern void real_set(long n, double lo, double hi, real *var);
extern void real_set_ramp(int numx, int numy, int numz, double lo,
                          double hi, real *var);
extern rcomplex *init_complex(long n, double lo, double hi);
extern rcomplex *init_complex_ramp(int numx, int numy, int numz,
                                   double lo, double hi);
extern void complex_set(long n, double lo, double hi, rcomplex *var);
extern void complex_set_ramp(int numx, int numy, int numz, double lo,
                             double hi, rcomplex *var);
extern long cache_adjust(long size);
extern void linear_real(int num, real *var, real low, real high);
extern void linear_rcomp(int num, rcomplex *var, rcomplex low, rcomplex high);

/* CACHE_LINE should be set to the size of a cache line in bytes */
#define CACHE_LINE 64

void c_print(rcomplex v);

extern void kernel_get_time(int nthr, double *tstart, double *tstop);
extern int kernel_get_nthr(void);
extern void start_omp_time(void);
extern void stop_omp_time(void);
extern void load_scalars(void);
extern void load_arrays(void);
