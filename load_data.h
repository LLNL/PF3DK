#define NCBUF 256

extern void load_scalars(void);
extern void load_arrays(void);

extern void load_arrays_omp45();

extern void load_scalars_omp45(int nx_in, int ny_in, int nxl_in,
                        int nyl_in, int nzl_in, int nxa_in, int nya_in,
                        int nza_in, int ngrd_in, long nplng_in, long ngtot_in, 
                        int num_teams_in, long ntheta_in, 
                        real clight_in, real csound_in, 
                        real dt_in, real dx_in, real dy_in, real dz_in,
                        real timunit_in,
                        int mp_rank_in, int mp_size_in,
                        int mp_p_in, int mp_q_in, int mp_r_in,
                        int mp_myp, int mp_myq, int mp_myr);
