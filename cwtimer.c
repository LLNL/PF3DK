#include <stdio.h>
#include <time.h>
#include "cwtimer.h"

#include <sys/time.h>

struct timeval time_zero;

void init_timer(void)
{
  int err;
  void *zone= 0;

  /* get the time at the start of the run */
  err= gettimeofday(&time_zero, zone);
}

double wsecond(double oldsec)
{
  /* gettimeofday is a Posix timer that returns a "microsecond clock". 
  */
  double tim;
  int err;
  struct timeval time;
  void *zone= 0;
  
  err= gettimeofday(&time, zone);
  tim= 1.0e-6*(time.tv_usec-time_zero.tv_usec)+(time.tv_sec-time_zero.tv_sec);
  return tim;
}
