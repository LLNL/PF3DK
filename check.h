typedef struct {
  char   *namkern;
  int     typedat; /* type 1 means a complex number (two real values).
                      type 0 is a single real value. */
  int     numvars; /* The number of variables that have reference values.
                      This is one for many tests, but couple5 has 4
                      complex output values. */
  int     numvals;  /* This is the number of values per variable.
                       It is 3 as of Dec. 5, 2019 but might increase
                       in the future. */
  char    **namvars; /* An array of variable names, one per value */
  int     *ndx;    /* A set of three indices for each reference value.
                      There will be numvars*numvals index triples. */
  real   *maxvals; /* numvars real maximum reference values. */
  real   *vals;    /* numvars*numvals real (type=0) or complex real (type=1)
                      reference values. */
} refkern;

extern refkern **kernptrs; /* information from reference runs is read into
                              structs accessible from this pointer. */
extern int nkern;  /* Contains the number of kernels for which data was
                      supplied after reference data is read */

extern void getref(int maxkern);
extern void testref(char *kern, refkern *thekern, double tol);
extern void check_cmplx(char *kern, char *vname, rcomplex *var, rcomplex *ref,
                        int *ndx, int nchek);
extern void show_cmplx(char *vname, rcomplex *var, int *ndx, int nchek);
extern void show_cmplx_dns(char *vname, rcomplex *var, int *ndx, int nchek);
extern void check_cmplx_hyd(char *kern, char *vname, rcomplex *var,
                            rcomplex *ref, int *ndx, int nchek);
extern void show_cmplx_hyd(char *vname, rcomplex *var, int *ndx, int nchek);
extern void check_cmplx_big(char *kern, char *vname, rcomplex *var,
                            rcomplex *ref, int *ndx, int nchek);
extern void show_cmplx_big(char *vname, rcomplex *var, int *ndx, int nchek);
extern void check_real(char *kern, char *vname, real *var, real *ref,
                       int *ndx, int nchek);
extern void show_real(char *vname, real *var, int *ndx, int nchek);
extern void show_real_dns(char *vname, real *var, int *ndx, int nchek);

extern int chekndx[9];
extern int nchek;
