extern rcomplex *t0, *t2, *t0_sav, *t2_sav, *tN_new;
extern rcomplex *t0_big, *t0_big_sav;
extern real *thetb_sav, *thetbig;

extern long nbig;
extern int     nxy_mx;
extern int     num_thr;

extern double *rnd;
extern double tot_tim;

extern real *iptmp, *optmp;
extern rcomplex *sndbuf, *rcvbuf;
extern rcomplex *afftbuf;
extern double complex *tmp_dbcom;
extern rcomplex *bigbuf;
extern rcomplex *t0DevPtr;

extern int nrows, ncols, nxfull, nyfull, nzfull;
extern int chunksize_x, chunksize_y;

extern void parm_init(void);
extern double wsecond(double offset);
extern double second(double oldsec);
extern void *wmalloc(size_t nbytes);

extern rcomplex *make_wave(double tvar_lo, double tvar_hi);
extern void free_wave(rcomplex *tvar);
extern void copy_wave(rcomplex *tvar, rcomplex *tvar_sav);
extern void t0_fixup(void);
extern void copy_arr2d(real *arr_old, real *arr_new);
extern void copy_carr2d(rcomplex *arr_old, rcomplex *arr_new);
extern void copy_arr3d(real *arr_old, real *arr_new);
extern void copy_carr3d(rcomplex *arr_old, rcomplex *arr_new);
extern void copy_to_tN(rcomplex * restrict tN, rcomplex * restrict tN_new);
extern void copy_from_tN(rcomplex * restrict tN_new, rcomplex * restrict tN);
extern void do_init(int nxl_in, int nyl_in, int nzl_in, int nthr_in);
extern void do_cleanup(void);
extern void reset_light(void);
extern void reset_thetb(void);
extern void reset_damp(void);
extern int  max_threads(void);
extern void getref(int maxnum);


#ifdef _OPENMP
#pragma omp declare target
#endif
void copy_tN_pre(rcomplex * restrict tN, rcomplex * restrict tN_new);
void copy_to_tN_omp45(rcomplex * restrict tvar, rcomplex * restrict tN_new);
void copy_from_tN_omp45(rcomplex * restrict tN_new, rcomplex * restrict tvar);
#ifdef _OPENMP
#pragma omp end declare target
#endif
