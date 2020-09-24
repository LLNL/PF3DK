/*  ************************************************

      SECOND= Cumulative CPU time for job in seconds.  MKS unit is seconds.
              Clock resolution should be less than 2% of Kernel 11 run-time.
              ONLY CPU time should be measured, NO system or I/O time included.
              In VM systems, page-fault time must be avoided (Direction 8).
              SECOND accuracy may be tested by calling: CALIBR test.
    
      Defing CLOCK_FUNCTION and/or BSD_TIMER on the compile line will
      often produce a working timer routine. If not, you will need to
      write your own. In addition, you should consider modifying 
      the timer routine if your system has a higher
      resolution clock. The run time for the loops is shorter
      when using a high resolution clock.
    
      Timing routines with resolutions of order 1/60 second are typical on 
      PCs and workstations, and require the use
      of multiple-pass loops around each kernel to make the run time
      at least 50 times the tick-period of the timing routine.
      Function SECOVT measures the overhead time for a call to SECOND.
    
      An independent calibration of the running time may be wise.
      Compare the Total Job Cpu Time printout at end of the LFK output file
      with the job Cpu time charged by your operating system, if available.
      The output written during the run provides a chance to check the
      timing with a stopwatch.
    
                USE THE HIGHEST RESOLUTION CPU-TIMER FUNCTION AVAILABLE
    ***********************************************************************
*/

#ifdef TIMEOFDAY
#define CLOCK_FUNCTION time_of_day
#define TICKS_PER_SEC 1
extern double time_of_day(void);
#endif

#ifdef _WIN32
#include <time.h>
#include <windows.h>
extern double get_time(void);
#define CLOCK_FUNCTION get_time
#define TICKS_PER_SEC 1
/* Defining USE_FANCY_CLOCK builds a version with a timer that is very 
   accurate on single processor Windows 95/98/NT computers, but it 
   gives incorrect answers on multi-processor NT systems. */
#define USE_FANCY_CLOCK
#endif

#ifdef ANSI_TIMER
#define CLOCK_FUNCTION clock
#endif

extern double second(double oldsec);

#ifndef CLOCK_FUNCTION
#ifdef BSD_TIMER
/* This will work on our Suns -- other BSD machines may differ?
    sys/time.h includes time.h on Sun */
#include <sys/time.h>
#include <sys/resource.h>
double second(double oldsec)
{
  double cpu, sys;
  struct rusage cpuTime;

  getrusage(RUSAGE_SELF, &cpuTime);
  cpu= cpuTime.ru_utime.tv_sec + 1.0e-6*cpuTime.ru_utime.tv_usec;
  sys= cpuTime.ru_stime.tv_sec + 1.0e-6*cpuTime.ru_stime.tv_usec;
  return cpu+sys-oldsec;
}

#else        /* Not BSD_TIMER */
/* Assume POSIX 1003.1-1990 standard timing interface.
   However-- CLK_TCK is noted as "obsolescent" there...  */
#include <time.h>
#include <sys/times.h>
/* Try to handle modest deviations from POSIX standard (e.g.- Sun).  */
#ifndef CLK_TCK
#include <unistd.h>
#ifndef CLK_TCK
#define CLK_TCK sysconf(_SC_CLK_TCK)
#endif
#endif
double second(double oldsec)
{
  static double ticksPerSecond= 0.0;
  struct tms cpuTime;
  long wallTicks= times(&cpuTime);
  double cpu, sys;

  if (ticksPerSecond==0.0) ticksPerSecond= CLK_TCK;
  cpu= cpuTime.tms_utime/(double)ticksPerSecond;
  sys= cpuTime.tms_stime/(double)ticksPerSecond;
  return cpu+sys-oldsec;
}

#endif    /* end of non BSD_TIMER branch */
#else     /* Branch where CLOCK_FUNCTION is defined */
/* Define CLOCK_FUNCTION then modify the following version to suit
   your architecture.  See sysdep.h for other suggestions.
 */
#include <time.h>
double second(double oldsec)
{
  return ((double)CLOCK_FUNCTION()) / ((double)TICKS_PER_SEC) - oldsec;
}
#endif

#ifdef MAC_CLOCK
double get_time(void)
{
  /* WARNING: Mac doesn't distinguish between system and user time
     (so count all time as user time) and it is totally wrong 
     if MultiFinder makes a switch while the program is being timed. 
  */
  double tim;
  UnsignedWide microTickCount;

  Microseconds(&microTickCount);
  tim= 1.0e-6*microTickCount.lo+1.0e-6*1024.0*1024.0*4096.0*microTickCount.hi;
  return tim;
}
#endif  /* MAC_CLOCK */

#ifdef _WIN32
/* WARNING: Use low resolution timer on a multi-processor Intel computer. */
double get_time(void)
{
	/* WARNING: PC doesn't distinguish between system and user time
	   (so count all time as user time).
	*/
#ifdef USE_FANCY_CLOCK
	unsigned _int64 time64, ticksPerSecond;

	QueryPerformanceCounter( (LARGE_INTEGER *)&time64);
	QueryPerformanceFrequency( (LARGE_INTEGER *)&ticksPerSecond);
	return ((double)(_int64)time64)/(_int64)ticksPerSecond;
#else
	return (double)clock()/CLOCKS_PER_SEC;
#endif
}
#endif   /* _WIN32 */

#ifdef TIMEOFDAY
#include <sys/time.h>
double time_of_day(void)
{
  /* gettimeofday is a Posix timer that returns a "microsecond clock". 
  */
  double tim;
  int err;
  struct timeval time;
  void *zone= 0;
  
  err= gettimeofday(&time, zone);
  tim= 1.0e-6*time.tv_usec+time.tv_sec;
  return tim;
}
#endif  /* TIMEOFDAY */
