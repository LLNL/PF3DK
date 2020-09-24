# This makes pf3d-fft-test
#
#________________________________________________________________________
#
# TIMER is the timer option appropriate to your system.
# You could use one of the examples in ctimer.c. 
# Use a more accurate timer if one is available on your system.
TIMER = -DSYSV_TIMER
#________________________________________________________________________
#
#
CODE_NAME = pf3dtest
#
# If you need any special load flags put them here...
#
# Default make rules for C - use full optimization
# Use a specific dependency line for any files that need lower optimization
.c.o :
	$(CC) $(CFLAGS) $(COPTIMIZE) -c $<

HEADERS = mytypes.h light.h lecuyer.h runparm.h pf3dbench.h \
          load_data.h check.h pf3d_fft.o

OBJS= pf3dbench.o ctimer.o cwtimer.o lecuyer.o ranfmp.o util.o init.o \
      load_data.o rotth.o check.o fft_func.o fft_util.o \
      couple_waves.o

OBJS_OMP45= load_data_omp45.o rotth_omp45.o couple_waves_omp45.o \
            lecuyer_omp45.o cufft.o fft_util_omp45.o

# This is the rule that make runs by default when you type make.
all: $(CODE_NAME)

# This is the rule to build a code with OpenMP 4.5 target offload functions
OMP45:  $(OBJS_OMP45) $(OBJS) $(HEADERS)
	$(COMP45) $(CFLAGS) $(COPTIMIZE) -DOMP45 -o $(SYS_TYPE)/$(CODE_NAME) $(OBJS) $(OBJS_OMP45) $(LDFLAGS)

#
#       benchmark containing kernels from pf3d.
$(CODE_NAME): $(OBJS) $(HEADERS)
	mkdir -p $(SYS_TYPE)
	$(CC) $(CFLAGS) $(COPTIMIZE) -o $(SYS_TYPE)/$(CODE_NAME) $(OBJS) $(LDFLAGS)
#
pf3dbench.o : pf3dbench.c $(HEADERS)
	$(CC) ${CFLAGS} -c pf3dbench.c

check.o : check.c $(HEADERS)
	$(CC) ${CFLAGS} -c check.c

cwtimer.o : cwtimer.c $(HEADERS)
	$(CC) ${CFLAGS} -c cwtimer.c

ctimer.o : ctimer.c $(HEADERS)
	$(CC) ${CFLAGS} -c ctimer.c

init.o : init.c $(HEADERS)
	$(CC) $(CFLAGS) -c init.c

lecuyer.o : lecuyer.c $(HEADERS)
	$(CC) $(CFLAGS) -c lecuyer.c

util.o : util.c $(HEADERS)
	$(CC) $(CFLAGS) -c util.c

# NOTE - most functions on the GPU will be optimized
# COPT_OMP45 is kept separate to deal with compiler bugs.

rotth_omp45.o : rotth_omp45.c $(HEADERS)
	$(COMP45_LO) ${CFLAGS} ${COPT_OMP45} -c rotth_omp45.c

load_data_omp45.o : load_data_omp45.c $(HEADERS)
	$(COMP45) ${CFLAGS} ${COPT_OMP45} -c load_data_omp45.c

lecuyer_omp45.o : lecuyer_omp45.c $(HEADERS)
	$(COMP45) ${CFLAGS} ${COPT_OMP45} -c lecuyer_omp45.c

pf3d_fft_omp45.o : pf3d_fft_omp45.c ${HEADERS}
	$(COMP45) ${CFLAGS} ${COPT_OMP45} -c pf3d_fft_omp45.c
#
# Clean up the junk, but leave the results...
clean:
	rm -f *.o ${CODE_NAME} *core *.dat *.optrpt *.lst
