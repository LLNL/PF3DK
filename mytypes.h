/* mytypes.h */

#ifndef __MYTYPES_H__

#include <stddef.h>
#include <stdlib.h>
#include <math.h>
/* Using C99's builtin complex instead of a complex struct */
#include <complex.h>

#ifdef USE_DOUBLE
typedef double real;
typedef complex double rcomplex;
#define RABS fabs
#else
typedef float real;
typedef complex float rcomplex;
#define RABS fabsf
#endif
typedef complex double dcomplex;

#define USE_DOUBLEFFT
#ifdef USE_DOUBLEFFT
typedef complex double fftcomplex;
#else
typedef complex float fftcomplex;
#endif

#ifdef __AVX512F__
    #ifdef USE_DOUBLE
    	#define real_lane_count 8
    #else
    	#define real_lane_count 16
    #endif
#else
    #ifdef USE_DOUBLE
    	#define real_lane_count 4
    #else
    	#define real_lane_count 8
    #endif
#endif


#ifdef __GNUC__
#define INLINE __always_inline
#else
#define INLINE inline
#endif

#define USE_RESTRICT
/* #undef USE_RESTRICT */
#ifdef USE_RESTRICT
#define RESTRICT restrict
#else
#define RESTRICT 
#endif

/* don't use "collapse" unless building an openMP code */
#ifdef _OPENMP
#define COLLAPSE(n) collapse(n)
#else
#define COLLAPSE(n)
#endif

typedef long integer;

/*******************************************************************************
 *                                                               light structure
 *******************************************************************************/
rcomplex *t0;                 /* first light wave */
rcomplex *t2;                 /* second light wave */

rcomplex *denlw;              /* acoustic wave */

#define ROOT3     1.732050807568877
#define TWOTHIRDS 0.66666666666666667
#ifndef PI
#define PI        3.1415926535897932384626433832795029
#endif

#ifndef FALSE
#define FALSE     0
#define TRUE      1
#endif

#ifndef M_PI
#define     M_PI            3.1415926535897932384626433832795029
#endif

#undef CONJ
#undef CREAL
#undef CIMAG

#ifdef USE_DOUBLE
#define SQRT(x) sqrt(x)
#define SIN(x) sin(x)
#define COS(x) cos(x)
#define EXP(x) exp(x)
#define IREAL 1.0i
#define CONJ(x) conj(x)
#define CREAL(x) creal(x)
#define CIMAG(x) cimag(x)
#define ZERO 0.0
#define QRTR 0.25
#define HALF 0.5
#define ONE 1.0
#define TWO 2.0
#define THREE 3.0
#define FOUR 4.0
#define EIGHT 8.0
#define e_mass 511.0
#else
#define SQRT(x) sqrtf(x)
#define SIN(x) sinf(x)
#define COS(x) cosf(x)
#define EXP(x) expf(x)
#define IREAL 1.0fi
#define CONJ(x) conjf(x)
#define CREAL(x) crealf(x)
#define CIMAG(x) cimagf(x)
#define ZERO 0.0f
#define QRTR 0.25f
#define HALF 0.5f
#define ONE 1.0f
#define TWO 2.0f
#define THREE 3.0f
#define FOUR 4.0f
#define EIGHT 8.0f
#define e_mass 511.0f
#endif

/* override C99 function calls with macros using intrinsics on Power 8/9 host */
#ifdef COMPLEXMACRO
#undef CONJ
#undef CREAL
#undef CIMAG
#define CONJ(x) ( __real__ (x) - _Complex_I * __imag__ (x) )
#define CREAL(x) ( __real__ (x) )
#define CIMAG(x) ( __imag__ (x) )
#endif

#define min(a,b)    ( ( (a) < (b) ) ? (a) : (b) )
#define max(a,b)    ( ( (a) < (b) ) ? (b) : (a) )
#define sign(a,b)   ( ( (b) < 0 ) ? - RABS(a) : RABS(a) )
#define sqr(a)      ( (double)(a) * (double)(a) )
#define pow32(a)    ( (a) * SQRT(a) )
#define zabs2(z)    ( CREAL(z)*CREAL(z)+CIMAG(z)*CIMAG(z) )
#define zabs(z)     SQRT( zabs2(z) )

/******************************************************************************/
/* Accessor macros
 */

/* macros containing guard cells */
#define CELLNDX(i,j,k)  (((k)+ngrd)*nya + (j)+ngrd)*nxa + (i)+ngrd

/* macros without guard cells */
#define CELTNDX(i,j,k)  (((k)*(nyl+1) + (j))*(nxl+1) + (i))
#define CELTNDX2(i,j)   ((j)*nxl + (i))
#define CELTNDX3(i,j,k) (((k)*nyl + (j))*nxl + (i))
#define CELTNDX2big(i,j) ( (j)*nxfull + (i) )
#define CELTNDX3big(i,j,k) (((k)*nyfull + (j))*nxfull + (i))

/* macros without guard cells for FFT in x-direction */
#define CELTNDX_x(i,j,k)  (((k)*(nyfull+1) + (j))*(nxfull+1) + (i))
/* macros without guard cells for FFT in y-direction */
#define CELTNDX_y(i,j,k)  (((k)*(nxfull+1) + (j))*(nyfull+1) + (i))

#define __MYTYPES_H__

#endif
