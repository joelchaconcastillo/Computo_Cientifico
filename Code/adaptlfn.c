
/* 

==========================================================================

adaptlfn.c -- a program for doing variable-at-a-time adaptive MCMC

Copyright (c) 2006 by Jeffrey S. Rosenthal (jeff@math.toronto.edu).

Licensed for general copying, distribution and modification according to
the GNU General Public License (http://www.gnu.org/copyleft/gpl.html).

----------------------------------------------------

Compile with "cc adaptlfn.c -lm", then run with "a.out".

==========================================================================

*/


#include <stdio.h>
#include <math.h>
#include <sys/time.h>

/* #define NUMBATCHES 100000 */
#define NUMBATCHES 1000000

#define BATCHLENGTH 100

#define TARGCHOICE 3  /* 0=Cauchy, 1=Normal, 2=Uniform, 3=Bimodal */

#define UNIFRAD 100.0

#define LOGADAPTLENGTH 100

#define MAXNUMSAME 100

/* #define ADAPTSIGMAS true */
#define ADAPTSIGMAS true

/* #define ADAPTLOG true */
#define ADAPTLOG true

/* #define VERBOSE true */

/* #define SPEEDUP true */
#define SPEEDUP true

#define NUMLEVELS 2

#define K 500

/* #define a1 -1
#define a2 -1 */
#define a1 1.0
#define a2 1.0
#define b1 1.0
#define b2 1.0
#define mu0 0.0
#define s0 1.0

/* #define PRINTLENGTH 1000 */
#define PRINTLENGTH 1000

/* #define TARGACCEPT 0.234 */
/* #define TARGACCEPT 0.44 */
#define TARGACCEPT 0.1

#define PI 3.14159265

#define myinfinity 999999999.9

#define XFILE "adaptlfnx"
#define LFILE "adaptlfnl"
#define LS0FILE "adaptlfnls0"
#define LS1FILE "adaptlfnls1"

double drand48(), square(), absval(), RR(), RRinv(), actual();

int r[K];
double mm[K], sd[K];

main()

{

    int i,j,k,batchnum, ii, jj, kk, xspacing;
    int numit, accepted, acount;
    int cumits, cumacc, logcount, numsame;
    double cumrate, prevrate;
    double cumsqdist, prevsqdist, sqdistsum;
    int logadapts, logchoose;
    double x, newx, logsigma[NUMLEVELS];
    int jumpcount, jumplogcount;
    double sqdiff, logalpha, tmp;
    double pilogest, pilogsum, logval;
    double sqnorm(), normal(), targlogdens(), adaptamount();
    FILE *fpx, *fpl, *fpls0, *fpls1;
    double begintime, endtime, currtime;
    struct timeval tmptv;

    /* INITIALISATIONS. */
    seedrand();
    numit = cumits = cumacc = numsame = logcount = 0;
    sqdistsum = 0.0;
    prevsqdist = 999999.0;
    logadapts = logchoose = 0;
    jumplogcount = jumpcount = 0;
    for (i=0; i<NUMLEVELS; i++)
      logsigma[i] = 0.0;
    prevrate = 0.0;
    x = 1.0;

/*
printf("%f ... %f \n", targlogdensraw(0.5), targlogdensraw(1.5));
exit(0);
*/

    gettimeofday (&tmptv, (struct timezone *)NULL);
    begintime = 1.0 * tmptv.tv_sec + 0.000001 * tmptv.tv_usec;

    printf("\nBeginning lfn adaption run.\n");
    printf("Batch size = %d full scans.\n", BATCHLENGTH);
    printf("Printing to screen every %d batches.\n", PRINTLENGTH);
    printf("Doing %d batches in total.\n", NUMBATCHES);
    printf("\n");

    if ((fpx = fopen(XFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", XFILE);
    }
#ifdef ADAPTLOG
    if ((fpl = fopen(LFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", LFILE);
    }
#endif
    if ((fpls0 = fopen(LS0FILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", LS0FILE);
    }
    if ((fpls1 = fopen(LS1FILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", LS1FILE);
    }

    xspacing = NUMBATCHES / 10000;
    if (xspacing == 0)
        xspacing = 1;

    fprintf(fpx, "\nxvector <- c(%f", actual(x, logcount));
#ifdef ADAPTLOG
    fprintf(fpl, "\nlvector <- c(%d", logcount);
#endif
    fprintf(fpls0, "\nls0vector <- c(%f", logsigma[0]);
    fprintf(fpls1, "\nls1vector <- c(%f", logsigma[1]);

   /* MAIN ITERATIVE LOOP. */
    for (batchnum=1; batchnum<=NUMBATCHES; batchnum++) {

        acount = 0;

        for (ii=1; ii<=BATCHLENGTH; ii++) {

	  numit++;
	  cumits++;

	  newx = x + exp(logsigma[logcount]) * normal();
	  logalpha = targlogdens(newx, logcount) - targlogdens(x, logcount);
	  accepted = ( log(drand48()) < logalpha );
	  if (accepted) {
              sqdistsum = sqdistsum + square(actual(newx)-actual(x));
	      if (x*newx < 0) {
		/* We've jumped over the origin. */
	        jumpcount++;
		if (logcount==1)
		  jumplogcount++;
	      }
	      x = newx;
	      acount++;
              cumacc++;
	  }

	  logval = log(1.0+sqnorm(x));
	  pilogsum = pilogsum + logval;

        } /* End of ii for loop. */

	/* Update various estimates etc. */
	pilogest = pilogsum / numit;

#ifdef ADAPTSIGMAS
	/* DO THE LOGSIGMA ADAPTING. */
        if (acount > BATCHLENGTH * TARGACCEPT )
	      logsigma[logcount] = logsigma[logcount] + adaptamount(batchnum);
	else if (acount < BATCHLENGTH * TARGACCEPT )
	      logsigma[logcount] = logsigma[logcount] - adaptamount(batchnum);
        if (divisible(batchnum, xspacing)) {
          fprintf(fpls0, ", %f", logsigma[0]);
          fprintf(fpls1, ", %f", logsigma[1]);
        }
#endif

#ifdef ADAPTLOG
      if (divisible(batchnum, LOGADAPTLENGTH)) {
          /* DO THE LOGCOUNT ADAPTING. */
          cumrate = ((double)cumacc) / cumits;
          cumsqdist = sqdistsum / cumits;
          cumacc = cumits = 0;
	  sqdistsum = 0.0;
	  numsame++;
#ifdef VERBOSE
          printf("CONSIDERING: prevl=%d, prevsqdist=%.2f, cumsqdist=%.2f \n",
				logcount, prevsqdist, cumsqdist);
#endif
          logadapts++;
	  if ( (numsame >= MAXNUMSAME) ||
  	    /* (absval(cumrate-TARGACCEPT) > absval(prevrate-TARGACCEPT)) */
	    		(cumsqdist < prevsqdist)
							) {
	    /* Modify logcount. */
	    prevrate = cumrate;
	    prevsqdist = cumsqdist;
            numsame = 0;
	    if (logcount==0) {
	      logcount = 1;
	      /* x = log(1.0 + x); */
	      x = RR(x);
	    } else if (logcount==1) {
	      logcount = 0;
	      /* x = exp(x) - 1.0; */
	      x = RRinv(x);
	    }
          }
	  if (logcount==1)
	    logchoose++;
          fprintf(fpl, ", %d", logcount);
      }
#endif

      if (divisible(batchnum, xspacing)) {
	  /* Write values to files. */
          fprintf(fpx, ", %f", actual(x,logcount));
      }

      /* OUTPUT SOME VALUES TO SCREEN. */
      if (divisible(batchnum, PRINTLENGTH)) {
        fflush(fpx);
        fflush(NULL);
        printf("x=%.2f, logcount=%d, logsigma=%.2f; ",
				x, logcount, logsigma[logcount]);
        printf("logval=%.3f, pilogest=%.3f ", logval, pilogest );
        gettimeofday (&tmptv, (struct timezone *)NULL);
        currtime = 1.0 * tmptv.tv_sec + 0.000001 * tmptv.tv_usec;
        printf("(%d/%d, %.2f secs)\n", batchnum, NUMBATCHES,
						currtime-begintime);
     }

} /* End of batchnum for loop. */

fprintf(fpx, " )\n\n");
fprintf(fpl, " )\n\n");
fprintf(fpls0, " )\n\n");
fprintf(fpls1, " )\n\n");

fclose(fpx);
fclose(fpl);
fclose(fpls0);
fclose(fpls1);

gettimeofday (&tmptv, (struct timezone *)NULL);
endtime = 1.0 * tmptv.tv_sec + 0.000001 * tmptv.tv_usec;

printf("\nDone ... TARGCHOICE = %d.\n", TARGCHOICE);
printf("Batch size = %d full scans.\n", BATCHLENGTH);
printf("Did %d batches in total.\n", NUMBATCHES);
printf("Printed to screen every %d batches.\n", PRINTLENGTH);
printf("Updated logcount every %d batches.\n", LOGADAPTLENGTH);
printf("Chose logarithm on %d of %d tries (%.2f\%).\n",
			logchoose, logadapts, (100.0*logchoose/logadapts) );
printf("Used logarithm on %d of %d origin jumps (%.2f\%).\n",
		jumplogcount, jumpcount, (100.0*jumplogcount/jumpcount) );
printf("Ellapsed time = %.2f seconds.\n\n", endtime-begintime);

return(0);

}


/* TARGLOGDENS */
double targlogdens( double x, int logcount )
{
    double targlogdensraw();

    if (logcount==0)
	return( targlogdensraw(x) );
    else
	return( targlogdensraw( exp(absval(x))-1.0 ) + absval(x) );
}

double targlogdensraw( double x )
{
    if (TARGCHOICE==0) {
	return( -log(1.0+square(x)) );
    } else if (TARGCHOICE==1) {
	return( - 0.5 * square(x) );
    } else if (TARGCHOICE==2) {
	if( (x >= -UNIFRAD) && (x <= UNIFRAD) )
          return(1.0);
	else
          return(-myinfinity);
    } else if (TARGCHOICE==3) {
	  return( log( exp(-square(x-5.0)/2.0) + exp(-square(x+5.0)/2.0) ) );
    } else {
	exit(1);
    }
}


/* SQNORM */
double sqnorm( double x )
{
    return(square(x));
}


double sqnormvarcomp( double t[] )
{

    int ii;
    double ss;
    ss = 0.0;
    for (ii=0; ii<K; ii++)
	ss = ss + (t[ii]-mu0)*(t[ii]-mu0);
    return(ss);

    /* return( A ); */
    /* return( (t[0]-mu0)*(t[0]-mu0) ); */
}

/* MIN */
double min( double xx, double yy)
{
  if (xx<yy)
    return(xx);
  else
    return(yy);
}

/* IMIN */
int imin( int n1, int n2)
{
  if (n1<n2)
    return(n1);
  else
    return(n2);
}

/* IMAX */
int imax( int n1, int n2)
{
  if (n1>n2)
    return(n1);
  else
    return(n2);
}

/* ADAPTAMOUNT */
double adaptamount( int iteration )
{
    /* return( 1.0 / imax(100, sqrt(iteration)) ); */
    return( min( 0.01, 1.0 / sqrt((double)iteration) ) );
    /* return(0.01); */
}

/* SEEDRAND: SEED RANDOM NUMBER GENERATOR. */
seedrand()
{
    int i, seed;
    struct timeval tmptv;
    gettimeofday (&tmptv, (struct timezone *)NULL);
    /* seed = (int) (tmptv.tv_usec - 1000000 *
                (int) ( ((double)tmptv.tv_usec) / 1000000.0 ) ); */
    seed = (int) tmptv.tv_usec;
    srand48(seed);
    (void)drand48();  /* Spin it once. */
    return(0);
}


/* NORMAL:  return a standard normal random number. */
double normal()
{
    double R, theta, drand48();

    R = - log(drand48());
    theta = 2 * PI * drand48();

    return( sqrt(2*R) * cos(theta));
}


double RR(double xx)
{
    if (xx < 0)
      return( - log(1.0 - xx) );
    else
      return( log(1.0 + xx) );
}

double RRinv(double xx)
{
    if (xx < 0)
      return( - exp(-xx) + 1.0 );
    else
      return( exp(xx) - 1.0 );
}

double actual(double xx, int logcount)
{
    if (logcount==0)
      return( xx );
    else if (logcount==1)
      return( RRinv(xx) );
    else
      exit(1);
}

double square(double xx)
{
    return(xx*xx);
}

double absval(double xx)
{
  if (xx >= 0.0)
    return(xx);
  else
    return(-xx);
}


int divisible (int n1, int n2)
{
  return (n1 == n2 * (n1/n2));
}


