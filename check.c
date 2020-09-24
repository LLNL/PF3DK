#define	__USE_XOPEN2K8 1

#include <stdio.h>
#include <string.h>

#include "mytypes.h"
#include "runparm.h"
#include "util.h"
#include "pf3dbench.h"
#include "check.h"

size_t getline(char **lineptr, size_t *n, FILE *stream);

#define CELBNDX(i,j,k) (((k)*(nyl+2) + (j))*(nxl+2) + (i))

/* This file contains functions to read reference values from
   a text file */

int nkern= 0;
refkern **kernptrs;
refkern *tstkern= 0;
double ref_tol= 0.6e-4;

extern FILE *file_ref;

int ECHO_READ= 0;

void getref(int maxkern)
{
  int num, err;
  char namkern[120];
  int typedat, numvars, numvals, ix, iy, iz, ii, jj, nd;
  long ntmp;
  int *ndx;
  double val1, val2;
  real *vals, *maxvals;
  refkern *thekern;
  char *tmp, varnam[120], *lineptr, **namvars;
  FILE *fref;

#ifdef BUILD_REF
  /* do not try to read an old reference when building a new one */
  return;
#else
  /* allocate enough storage for maxkern reference results */
  if(mp_rank == 0) printf("\nReading reference file.\n");
  kernptrs= (refkern **)malloc(sizeof(refkern *)*maxkern);
  
#ifdef OMP45_BUILD
  fref= fopen("reference_vals_omp45.txt", "r");
#else
  fref= fopen("reference_vals.txt", "r");
#endif
  while(nkern < maxkern) {
    /* read the kernel name */
    err= fscanf(fref, "%s\n", namkern);
    if(err == EOF) break;
    if(ECHO_READ && mp_rank == 0) printf("nkern is %d\n", nkern);
    thekern= (refkern *)malloc(sizeof(refkern));
    kernptrs[nkern]= thekern;
    if(ECHO_READ && mp_rank == 0) printf("namkern is %s\n", namkern);
    tmp= (char *) malloc(strlen(namkern)+1);
    strcpy(tmp, namkern);
    thekern->namkern= tmp;
    /* read the data type */
    err= fscanf(fref, "%d", &typedat);
    if(ECHO_READ && mp_rank == 0) printf("data type is %d\n", typedat);
    thekern->typedat= typedat;
    err= fscanf(fref, "%d", &numvars);
    if(ECHO_READ && mp_rank == 0) printf("numvars is %d\n", numvars);
    thekern->numvars= numvars;
    err= fscanf(fref, "%d", &numvals);
    if(ECHO_READ && mp_rank == 0) printf("numvals is %d\n", numvals);
    thekern->numvals= numvals;
    ndx= (int *) malloc(sizeof(int)*3*numvars*numvals);
    thekern->ndx= ndx;
    maxvals= (real *) malloc(sizeof(real)*numvars);
    thekern->maxvals= maxvals;
    if(typedat == 0) {
      vals= (real *) malloc(sizeof(real)*numvars*numvals);
    } else {
      vals= (real *) malloc(sizeof(real)*2*numvars*numvals);
    }
    thekern->vals= vals;
    namvars= (char **) malloc(sizeof(char *)*numvars*numvals);
    thekern->namvars= namvars;
    for(jj= 0; jj < numvars; jj++) {
      err= fscanf(fref, "%s ,%le", varnam, &val1);
      if(ECHO_READ && mp_rank == 0) printf("name of maxvar is %s and max is %le\n", varnam, val1);
      maxvals[jj]= val1;
      for(ii= 0; ii < numvals; ii++) {
        nd= ii+jj*numvals;
        if(typedat == 0) {
          /* real variable, one per line */
          err= fscanf(fref, "%s ,%d,%d,%d,%le", varnam, &ix, &iy, &iz, &val1);
          tmp= (char *) malloc(strlen(varnam)+1);
          strcpy(tmp, varnam);
          namvars[nd]= tmp;
          ndx[3*nd]=   ix;
          ndx[3*nd+1]= iy;
          ndx[3*nd+2]= iz;
          vals[nd]= val1;
          if(ECHO_READ && mp_rank == 0) printf("retrieved data for kernel %s, variable %s\n", namkern, tmp);
        } else {
          /* complex variable, two reals per line */
          err= fscanf(fref, "%s ,%d,%d,%d,%le,%le", varnam, &ix, &iy, &iz, &val1, &val2);
          tmp= (char *) malloc(strlen(varnam)+1);
          strcpy(tmp, varnam);
          namvars[nd]= tmp;
          ndx[3*nd]=   ix;
          ndx[3*nd+1]= iy;
          ndx[3*nd+2]= iz;
          vals[2*nd]= val1;
          vals[2*nd+1]= val2;
        }
      }
      if(ECHO_READ && mp_rank == 0) printf("for kernel %d, the variable name is %s\n", ii, varnam);
    }
    nkern++;
  }
  if(mp_rank == 0) printf("Number of kernels with reference values is %d\n\n", nkern);
#endif
}

void testref(char *kern, refkern *thekern, double tol)
{
  /* This function compares results for kernel "kern" to the reference
     values for that kernel */
  refkern *kernref;
  int i, ik, j, k, nd, refid, nchek;
  int typ, typref, numvars, numvarsref, numvals, numvalsref;
  int ix, iy, iz, ixref, iyref, izref, ndat, ndatref;
  int *ndx, *ndxref;
  real *vals, *valsref, *maxvalsref;
  double fuzz= 1.0e-10;
  double df, dfr, dfi, df2;
  char **namvars, **namvarsref, *nam, *thevar;

  /* loop through all reference values looking for the specified kernel */
  /* WARNING - this code assumes that only one variable is passed in via
     thekern. */

#ifdef BUILD_REF
  /* do not check against a reference if in the process of building one */
  return;
#endif
  
  refid= -1;
  for(ik= 0; ik < nkern; ik++) {
    kernref= kernptrs[ik];
    nam= kernref->namkern;
    /* record the index where the reference kernel has the requested name */
    if(!strcmp(nam, kern)) {
      refid= ik;
      break;
    }
  }
  if(refid < 0) {
    if(mp_rank == 0) printf("ERROR: kernel %s not found \n", kern);
    return;
  }
  /* Extract fields from this kernel and the reference.
   Validate as necessary. */
  typ= thekern->typedat;
  typref= kernref->typedat;
  if( typ != typref) {
    if(mp_rank == 0) printf("ERROR: data type does not match for kernel %s\n", kern);
    return;
  }
  numvars= thekern->numvars;
  numvarsref= kernref->numvars;
  if( numvars != 1) {
    if(mp_rank == 0) printf(" WARNING - testref only expects one input var\n");
  }
  numvals= thekern->numvals;
  numvalsref= kernref->numvals;
  if( numvals != numvalsref) {
    if(mp_rank == 0) printf("ERROR: numvals does not match for kernel %s\n", kern);
  }
  namvars= thekern->namvars;
  namvarsref= kernref->namvars;
  ndx= thekern->ndx;
  ndxref= kernref->ndx;
  maxvalsref= kernref->maxvals;
  vals= thekern->vals;
  valsref= kernref->vals;
  ndat= numvars*numvals;
  ndatref= numvarsref*numvalsref;
  thevar= namvars[0]; /* only one variable name passed */
  for(i= 0; i < ndat; i++) {
    ix= ndx[3*i];
    iy= ndx[3*i+1];
    iz= ndx[3*i+2];
    for(j= 0; j < numvarsref; j++) {
      for(k= 0; k < numvalsref; k++) {
        nd= k+j*numvalsref;
        if( !strcmp(thevar, namvarsref[nd] )) {
          /* found one of the reference values for the
             desired variable */
          ixref= ndxref[3*nd];
          iyref= ndxref[3*nd+1];
          izref= ndxref[3*nd+2];
          if(ix == ixref && iy == iyref && iz == izref) {
            /* this is the matching reference entry */
            if(typ == 0) {
              /* compare real values */
              df= vals[i]-valsref[nd];
              df2= sqrt( (df*df) / (valsref[nd]*valsref[nd]
                                    +fuzz*maxvalsref[j]*maxvalsref[j]) );
            } else {
              /* compare complex values */
              dfr= vals[2*i]-valsref[2*nd];
              dfi= vals[2*i+1]-valsref[2*nd+1];
              df2= sqrt( (dfr*dfr+dfi*dfi) / (valsref[2*nd]*(double)valsref[2*nd]+
                                              valsref[2*nd+1]*(double)valsref[2*nd+1]
                                              +fuzz*maxvalsref[j]) );
            }
            /* The != test is intended to detect NANs *. */
            if(isnan(df2) || isinf(df2) || df2 > tol) {
              if(mp_rank == 0) printf("ERROR:kernel %s:var=%s(%d,%d,%d) has diff %e > tolerance %e, maxval is %e\n",
                     kern, thevar, ix, iy, iz, df2, tol, maxvalsref[j]);
            } else {
              if(mp_rank == 0) printf("kernel %s difference %le is within tolerance %e\n", kern, df2, tol);
            }
          }   /* done with matching (ix,iy,iz) */       
        }
        break;
      }
    }
  }
}

void check_cmplx(char *kern, char *vname, rcomplex *var, rcomplex *ref, int *ndx, int nchek)
{
  double sumsq, difbig, difsq, difrms, maxabs2, maxval;
  double varr, vari, refr, refi, difr, difi;
  real vals[nchek*2];
  int i, ix, iy, iz, ii;
  char *nams[1];
  
  sumsq= 0.0;
  difbig= 0.0;
  maxabs2= 0.0;
  for (iz=0; iz<nzl; iz++) {
    for (iy=0; iy<nyl; iy++) {    
      for (ix=0; ix<nxl; ix++) {
        ii= CELTNDX(ix,iy,iz);
        varr= creal(var[ii]);
        vari= cimag(var[ii]);
        refr= creal(ref[ii]);
        refi= cimag(ref[ii]);
        difr= varr-refr;
        difi= vari-refi;
        difsq= difr*difr+difi*difi;
        sumsq= sumsq+difsq;
        if(difbig < difsq) difbig= difsq;
        maxval= varr*varr+vari*vari;
        if(maxval > maxabs2) maxabs2= maxval;
      }
    }
  }
  difrms= sqrt(sumsq/(nxl*nyl*nzl));
  difbig= sqrt(difbig);
  maxval= sqrt(maxabs2);
  printf("\n%s: RMS diff is %e, max diff is %e\n", vname, difrms, difbig);
#ifdef BUILD_REF
  fprintf(file_ref, "%s%s,%e\n", vname, " ", maxval );
#endif
  for(i= 0; i < nchek; i++) {
      ix= ndx[3*i];
      iy= ndx[3*i+1];
      iz= ndx[3*i+2];
      ii= CELTNDX(ix,iy,iz);
      printf("out %s(%d,%d,%d)= (%e, %e)\n", vname, ix,
             iy, iz, creal(var[ii]), cimag(var[ii]) );
      vals[2*i]=   creal(var[ii]);
      vals[2*i+1]= cimag(var[ii]);
#ifdef BUILD_REF
      fprintf(file_ref, "%s%s,%d,%d,%d,%e, %e\n", vname, " ", ix,
             iy, iz, creal(var[ii]), cimag(var[ii]) );
#endif
  }
  if(!tstkern) {
    tstkern= (refkern *)malloc(sizeof(refkern));
  }
  tstkern->namkern= kern;
  tstkern->numvals= nchek;
  tstkern->numvars= 1;
  tstkern->typedat= 1; /* compex variable */
  tstkern->ndx= ndx;
  tstkern->vals= vals;
  nams[0]= vname;
  tstkern->namvars= nams;
  testref(kern, tstkern, ref_tol);
}

void show_cmplx(char *vname, rcomplex *var, int *ndx, int nchek)
{
  int i, ix, iy, iz, ii;
  
  for(i= 0; i < nchek; i++) {
      ix= ndx[3*i];
      iy= ndx[3*i+1];
      iz= ndx[3*i+2];
      ii= CELTNDX(ix,iy,iz);
      printf("%s(%d,%d,%d)= (%e, %e)\n", vname, ix,
             iy, iz, creal(var[ii]), cimag(var[ii]) );
  }
}

void show_cmplx_dns(char *vname, rcomplex *var, int *ndx, int nchek)
{
  int i, ix, iy, iz, ii;
  
  for(i= 0; i < nchek; i++) {
      ix= ndx[3*i];
      iy= ndx[3*i+1];
      iz= ndx[3*i+2];
      ii= CELTNDX3(ix,iy,iz);
      printf("%s(%d,%d,%d)= (%e, %e)\n", vname, ix,
             iy, iz, creal(var[ii]), cimag(var[ii]) );
  }
}

void check_cmplx_hyd(char *kern, char *vname, rcomplex *var, rcomplex *ref, int *ndx, int nchek)
{
  double sumsq, difbig, difsq, difrms, maxabs2, maxval;
  double varr, vari, refr, refi, difr, difi;
  real vals[nchek*2];
  int i, ix, iy, iz, ii;
  char *nams[1];
  
  sumsq= 0.0;
  difbig= 0.0;
  maxabs2= 0.0;
  for (iz=0; iz<nzl; iz++) {
    for (iy=0; iy<nyl; iy++) {    
      for (ix=0; ix<nxl; ix++) {
        ii= CELBNDX(ix,iy,iz);
        varr= creal(var[ii]);
        vari= cimag(var[ii]);
        refr= creal(ref[ii]);
        refi= cimag(ref[ii]);
        difr= varr-refr;
        difi= vari-refi;
        difsq= difr*difr+difi*difi;
        sumsq= sumsq+difsq;
        if(difbig < difsq) difbig= difsq;
        maxval= varr*varr+vari*vari;
        if(maxval > maxabs2) maxabs2= maxval;
      }
    }
  }
  difrms= sqrt(sumsq/(nxl*nyl*nzl));
  difbig= sqrt(difbig);
  maxval= sqrt(maxabs2);
  printf("\n%s: RMS diff is %e, max diff is %e\n", vname, difrms, difbig);
#ifdef BUILD_REF
  fprintf(file_ref, "%s%s,%e\n", vname, " ", maxval );
#endif
  for(i= 0; i < nchek; i++) {
      ix= ndx[3*i];
      iy= ndx[3*i+1];
      iz= ndx[3*i+2];
      ii= CELBNDX(ix,iy,iz);
      printf("out %s(%d,%d,%d)= (%e, %e)\n", vname, ix,
             iy, iz, creal(var[ii]), cimag(var[ii]) );
      vals[2*i]=   creal(var[ii]);
      vals[2*i+1]= cimag(var[ii]);
#ifdef BUILD_REF
      fprintf(file_ref, "%s%s,%d,%d,%d,%e, %e\n", vname, " ", ix,
             iy, iz, creal(var[ii]), cimag(var[ii]) );
#endif
  }
  if(!tstkern) {
    tstkern= (refkern *)malloc(sizeof(refkern));
  }
  tstkern->namkern= kern;
  tstkern->numvals= nchek;
  tstkern->numvars= 1;
  tstkern->typedat= 1; /* compex variable */
  tstkern->ndx= ndx;
  tstkern->vals= vals;
  nams[0]= vname;
  tstkern->namvars= nams;
  testref(kern, tstkern, ref_tol);
}

void show_cmplx_hyd(char *vname, rcomplex *var, int *ndx, int nchek)
{
  int i, ix, iy, iz, ii;
  
  for(i= 0; i < nchek; i++) {
      ix= ndx[3*i];
      iy= ndx[3*i+1];
      iz= ndx[3*i+2];
      ii= CELBNDX(ix,iy,iz);
      printf("%s(%d,%d,%d)= (%e, %e)\n", vname, ix,
             iy, iz, creal(var[ii]), cimag(var[ii]) );
  }
}

void check_cmplx_big(char *kern, char *vname, rcomplex *var, rcomplex *ref, int *ndx, int nchek)
{
  double sumsq, difbig, difsq, difrms, maxabs2, maxval;
  double varr, vari, refr, refi, difr, difi;
  real vals[nchek*2];
  int i, ix, iy, iz, ii;
  char *nams[1];
  
  sumsq= 0.0;
  difbig= 0.0;
  maxabs2= 0.0;
  for (iz=0; iz<nzl; iz++) {
    for (iy=0; iy<nyl; iy++) {    
      for (ix=0; ix<nxl; ix++) {
        ii= CELTNDX3big(ix,iy,iz);
        varr= creal(var[ii]);
        vari= cimag(var[ii]);
        refr= creal(ref[ii]);
        refi= cimag(ref[ii]);
        difr= varr-refr;
        difi= vari-refi;
        difsq= difr*difr+difi*difi;
        sumsq= sumsq+difsq;
        if(difbig < difsq) difbig= difsq;
        maxval= varr*varr+vari*vari;
        if(maxval > maxabs2) maxabs2= maxval;
      }
    }
  }
  difrms= sqrt(sumsq/(nxl*nyl*nzl));
  difbig= sqrt(difbig);
  maxval= sqrt(maxabs2);
  printf("\n%s: RMS diff is %e, max diff is %e\n", vname, difrms, difbig);
#ifdef BUILD_REF
  fprintf(file_ref, "%s%s,%e\n", vname, " ", maxval );
#endif
  for(i= 0; i < nchek; i++) {
      ix= ndx[3*i];
      iy= ndx[3*i+1];
      iz= ndx[3*i+2];
      ii= CELTNDX3big(ix,iy,iz);
      printf("out %s(%d,%d,%d)= (%e, %e)\n", vname, ix,
             iy, iz, creal(var[ii]), cimag(var[ii]) );
      vals[2*i]=   creal(var[ii]);
      vals[2*i+1]= cimag(var[ii]);
#ifdef BUILD_REF
      fprintf(file_ref, "%s%s,%d,%d,%d,%e, %e\n", vname, " ", ix,
             iy, iz, creal(var[ii]), cimag(var[ii]) );
#endif
  }
  if(!tstkern) {
    tstkern= (refkern *)malloc(sizeof(refkern));
  }
  tstkern->namkern= kern;
  tstkern->numvals= nchek;
  tstkern->numvars= 1;
  tstkern->typedat= 1; /* compex variable */
  tstkern->ndx= ndx;
  tstkern->vals= vals;
  nams[0]= vname;
  tstkern->namvars= nams;
  testref(kern, tstkern, ref_tol);
}

void show_cmplx_big(char *vname, rcomplex *var, int *ndx, int nchek)
{
  int i, ix, iy, iz, ii;
  
  for(i= 0; i < nchek; i++) {
      ix= ndx[3*i];
      iy= ndx[3*i+1];
      iz= ndx[3*i+2];
      ii= CELTNDX3big(ix,iy,iz);
      printf("%s(%d,%d,%d)= (%e, %e)\n", vname, ix,
             iy, iz, creal(var[ii]), cimag(var[ii]) );
  }
}

void check_real(char *kern, char *vname, real *var, real *ref, int *ndx, int nchek)
{
  double sumsq, difbig, difsq, difrms, dif, maxabs2, maxval;
  real vals[nchek];
  int i, ix, iy, iz, ii;
  char *nams[1];
  
  sumsq= 0.0;
  difbig= 0.0;
  maxabs2= 0.0;
  for (iz=0; iz<nzl; iz++) {
    for (iy=0; iy<nyl; iy++) {    
      for (ix=0; ix<nxl; ix++) {
        ii= CELTNDX(ix,iy,iz);
        dif= var[ii]-ref[ii];
        difsq= dif*dif;
        sumsq= sumsq+difsq;
        if(difbig < difsq) difbig= difsq;
        maxval= var[ii]*(double)var[ii];
        if(maxval > maxabs2) maxabs2= maxval;
      }
    }
  }
  difrms= sqrt(sumsq/(nxl*nyl*nzl));
  difbig= sqrt(difbig);
  maxval= sqrt(maxabs2);
  printf("\n%s: RMS diff is %e, max diff is %e\n", vname, difrms, difbig);
#ifdef BUILD_REF
  fprintf(file_ref, "%s%s,%e\n", vname, " ", maxval );
#endif
  for(i= 0; i < nchek; i++) {
      ix= ndx[3*i];
      iy= ndx[3*i+1];
      iz= ndx[3*i+2];
      ii= CELTNDX(ix,iy,iz);
      printf("out %s(%d,%d,%d)= %e\n", vname, ix, iy, iz, var[ii] );
      vals[i]= var[ii];
#ifdef BUILD_REF
      fprintf(file_ref, "%s%s,%d,%d,%d,%e\n", vname, " ", ix, iy, iz, var[ii] );
#endif
  }
  if(!tstkern) {
    tstkern= (refkern *)malloc(sizeof(refkern));
  }
  tstkern->namkern= kern;
  tstkern->numvals= nchek;
  tstkern->numvars= 1;
  tstkern->typedat= 0; /* real variable */
  tstkern->ndx= ndx;
  tstkern->vals= vals;
  nams[0]= vname;
  tstkern->namvars= nams;
  testref(kern, tstkern, ref_tol);
}

void show_real(char *vname, real *var, int *ndx, int nchek)
{
  int i, ix, iy, iz, ii;
  
  for(i= 0; i < nchek; i++) {
      ix= ndx[3*i];
      iy= ndx[3*i+1];
      iz= ndx[3*i+2];
      ii= CELTNDX(ix,iy,iz);
      printf("%s(%d,%d,%d)= %e\n", vname, ix, iy, iz, var[ii] );
  }
}

void show_real_dns(char *vname, real *var, int *ndx, int nchek)
{
  int i, ix, iy, iz, ii;
  
  for(i= 0; i < nchek; i++) {
      ix= ndx[3*i];
      iy= ndx[3*i+1];
      iz= ndx[3*i+2];
      ii= CELTNDX3(ix,iy,iz);
      printf("%s(%d,%d,%d)= %e\n", vname, ix, iy, iz, var[ii] );
  }
}
