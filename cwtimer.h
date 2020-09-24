#include <stdlib.h>
#include "time.h"

typedef struct timespec TMSPEC;

void comp_walltime(TMSPEC start, TMSPEC finish);
double get_walltime(TMSPEC start);
double wsecond(double oldsec);
void init_timer(void);
void init_clock(struct timespec *start);
