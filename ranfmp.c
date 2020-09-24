/* random number generator copied from xtr/ranf.c
 * usage :
 * iseed=123456789;
 * x=ranfmp(&iseed); 0 <= x < 1
 * irseed = ulmodm(rand_advk(k),irseed); -> simulate k calls to ranf
 * (cf main() at the end of this file)
 */



long ulmodm(long a, long b){
  /* return (a*b) mod m where m=2^31-1
   * a and b are 32 bit long integer
   */
  long m = 2147483647;          /* m = 2^31 - 1 */
  long twop16=65536;
  unsigned long ahi, alo, bhi, blo, q1, r;
  ahi=a/twop16;
  alo=a-ahi*twop16;
  bhi=b/twop16;
  blo=b-bhi*twop16;
  q1=ahi*bhi;
   r=2*(q1%m); /* 2^32 mod m = 2 */
  q1=(alo*bhi+ahi*blo)%m;
  r=r%m + ((2*(q1/twop16))%m + ((q1%twop16)*twop16)%m)%m;
  q1=(alo*blo)%m;
  r=r%m + q1;
  r=r%m;
  return (long)(r);
} 



long rand_advk(long k){
  /* advance k step in the random number list
   * (i.e. act like k calls to ranf)
   */
  long m = 2147483647;          /* m = 2^31 - 1 */
  long a=16807;                 /* 7^5 */
  long r, q;
  long anmodm[65];
  int i;
  anmodm[1]=a;
  for(i=2;i<=64;i++)
    anmodm[i]=ulmodm(anmodm[i-1],anmodm[i-1]);
  r=1;
  for(i=1;i<=64;i++){
    q=k/2;
    if (k - 2*q) r=ulmodm(r,anmodm[i]);
    k=q;
  }
  return (r);
}



long irand_bis(long iseed)
{
  long a = 16807;               /* a = 7^5 */
  long m = 2147483647;          /* m = 2^31 - 1 */
  long q = 127773;              /* q = m % a */
  long r = 2836;                /* r = mod(m,a) */

  long seed, hi, lo;

  /*  
     *  Algorithm replaces seed by mod(a*seed,m). First represent seed = q*hi + lo.
     *  Then a*seed = a*q*hi + lo = (m - r)*hi + a*lo = (a*lo - r*hi) + m*hi,
     *  and new seed = a*lo - r*hi unless negative; if so, then add m.
     */
  seed = iseed;
  hi = seed/q;
  lo = seed - q*hi;
  seed = a*lo - r*hi;
  /* "seed" will always be a legal integer of 32 bits (including sign). */
  if (seed < 0)
    seed = seed + m;

  return(seed);
}

double ranfmp(long *irfseed)
{
  double res = ((double) *irfseed) / 2147483648.0;
  *irfseed = irand_bis(*irfseed);
  return(res);
}
