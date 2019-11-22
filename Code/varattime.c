
/* 

==========================================================================

varattime.c -- a program for doing variable-at-a-time adaptive MCMC

Copyright (c) 2006 by Jeffrey S. Rosenthal (jeff@math.toronto.edu).

Licensed for general copying, distribution and modification according to
the GNU General Public License (http://www.gnu.org/copyleft/gpl.html).

----------------------------------------------------

Compile with "cc varattime.c -lm", then run with "a.out".

==========================================================================

*/


#include <stdio.h>
#include <math.h>
#include <sys/time.h>

/* #define NUMBATCHES 100000 */
#define NUMBATCHES 5000

/* #define EMPBAYES true */

#define BATCHLENGTH 50

/* Ellapsed time = 59950.69 seconds = 16.65 hours. */

/* #define ADAPTSIGMAS true */
#define ADAPTSIGMAS true

/* #define VERBOSE true */

/* #define HYPERVERBOSE true */

/* #define SPEEDUP true */
#define SPEEDUP true

#define K 500

/* #define a1 -1
#define a2 -1 */
#define a1 1.0
#define a2 1.0
#define b1 1.0
#define b2 1.0
#define mu0 0.0
#define s0 1.0

#define PRINTLENGTH 1

#define TARGACCEPT 0.44
/* #define TARGACCEPT 0.234 */

#define PI 3.14159265

#define X0FILE "adaptvaratx0"
#define X1FILE "adaptvaratx1"
#define X2FILE "adaptvaratx2"
#define MUFILE "adaptvaratmu"
#define AFILE "adaptvaratA"
#define VFILE "adaptvaratV"
#define LS0FILE "adaptvaratls0"
#define LS1FILE "adaptvaratls1"
#define LS2FILE "adaptvaratls2"
#define LSMUFILE "adaptvaratlsmu"
#define LSAFILE "adaptvaratlsA"
#define LSVFILE "adaptvaratlsV"
#define EFILE "adaptvarate"
#define MASTERFILE "adaptvaratm"

double drand48();

int r[K];
double mm[K], sd[K];

#define JMAX 1000

double Y[K][JMAX];

main()

{

    int i,j,k,batchnum, ii, jj, kk, xspacing;
    int numit, accepted, acount[K+3];
    double t[K+3];
    double newt[K+3];
    double logsigma[K+3];
    double sqdiff, logalpha, tmp;
    double pilogest, pilogsum, logval;
    double sqnorm(), normal(), targlogdens(), targlogdenscomp(), adaptamount();
    double square();
    FILE *fpx0, *fpx1, *fpx2, *fpmu, *fpA, *fpV, *fpe, *fpm;
    FILE *fpls0, *fpls1, *fpls2, *fplsmu, *fplsA, *fplsV;
    double begintime, endtime, currtime;
    struct timeval tmptv;

    /* DATA VALUES */
    seedrand();
    for (jj=0; jj<K/10; jj++) {
      r[10*jj + 0] = 5;
      r[10*jj + 1] = 50;
      r[10*jj + 2] = 500;
      r[10*jj + 3] = r[10*jj + 4] = r[10*jj + 5] = 5;
      r[10*jj + 6] = r[10*jj + 7] = 50;
      r[10*jj + 8] = r[10*jj + 9] = 500;
    }
    for (ii=0; ii<K; ii++) {
      mm[ii] = 1.0 * ii;
      sd[ii] = 10.0;
      tmp = 0.0;
      for (jj=0; jj<r[ii]; jj++) {
        Y[ii][jj] = mm[ii] + normal() * sd[ii];
        tmp = tmp + Y[ii][jj];
#ifdef VERBOSE
        printf("Y[%d][%d] = %f \n", ii, jj, Y[ii][jj]);
#endif
      }
      printf("Mean of Y[%d][.] = %f;   ", ii, tmp/r[ii]);
    }

#ifdef VERBOSE
  t[0] = t[1] = t[2] = t[3] = t[4] = t[5] = 0.0;
  printf("t's = %f ... targlogdens = %f \n\n", t[0], targlogdens(t) );
  t[0] = 10.0;
  printf("big t[0] ... targlogdens = %f \n\n", targlogdens(t) );
  t[0] = 0.0;
  t[1] = 10.0;
  printf("big t[1] ... targlogdens = %f \n\n", targlogdens(t) );
  t[1] = 0.0;
  t[2] = 10.0;
  printf("big t[2] ... targlogdens = %f \n\n", targlogdens(t) );
  t[0] = t[1] = t[2] = t[3] = t[4] = t[5] = 10.0;
  printf("t's = %f ... targlogdens = %f \n\n", t[0], targlogdens(t) );
  t[0] = t[1] = t[2] = t[3] = t[4] = t[5] = 100.0;
  printf("t's = %f ... targlogdens = %f \n\n", t[0], targlogdens(t) );
  t[0] = t[1] = t[2] = t[3] = t[4] = t[5] = 1000.0;
  printf("t's = %f ... targlogdens = %f \n\n", t[0], targlogdens(t) );
#endif


    /* INITIALISATIONS. */
    numit = 0;
    /* mu = 0.0; */
    /* A = V = 1.0; */
    for (i=0; i<K+3; i++)
	t[i] = 0.0;
#ifdef EMPBAYES
    t[K+2] = 0.00434;
#endif
    for (i=0; i<K+3; i++)
	logsigma[i] = 0.0;
    logval = pilogest = pilogsum = 0.0;
    for (kk=0; kk<K+3; kk++)
	newt[kk] = t[kk];

    gettimeofday (&tmptv, (struct timezone *)NULL);
    begintime = 1.0 * tmptv.tv_sec + 0.000001 * tmptv.tv_usec;

    printf("\nBeginning variable-at-time adaption run.\n");
    printf("K = %d;  total number of variables = %d.\n", K, K+3);
    printf("Batch size = %d full scans.\n", BATCHLENGTH);
    printf("Printing to screen every %d batches.\n", PRINTLENGTH);
    printf("Doing %d batches in total.\n", NUMBATCHES);
    for (kk=0; kk<K; kk++)
      printf("r[%d]=%d, ", kk, r[kk]);
    printf("\n\n");

    if ((fpx0 = fopen(X0FILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", X0FILE);
    }
    if ((fpx1 = fopen(X1FILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", X1FILE);
    }
    if ((fpx2 = fopen(X2FILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", X2FILE);
    }
    if ((fpmu = fopen(MUFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", MUFILE);
    }
    if ((fpA = fopen(AFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", AFILE);
    }
    if ((fpV = fopen(VFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", VFILE);
    }

    if ((fpls0 = fopen(LS0FILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", LS0FILE);
    }
    if ((fpls1 = fopen(LS1FILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", LS1FILE);
    }
    if ((fpls2 = fopen(LS2FILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", LS2FILE);
    }
    if ((fplsmu = fopen(LSMUFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", LSMUFILE);
    }
    if ((fplsA = fopen(LSAFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", LSAFILE);
    }
    if ((fplsV = fopen(LSVFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", LSVFILE);
    }

    if ((fpe = fopen(EFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", EFILE);
    }
    if ((fpm = fopen(MASTERFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", MASTERFILE);
    }

    xspacing = NUMBATCHES / 10000;
    if (xspacing == 0)
        xspacing = 1;

    fprintf(fpx0, "\nx0vector <- c(%f", t[0]);
    fprintf(fpx1, "\nx1vector <- c(%f", t[1]);
    fprintf(fpx2, "\nx2vector <- c(%f", t[2]);
    fprintf(fpmu, "\nmuvector <- c(%f", t[K]);
    fprintf(fpA, "\nAvector <- c(%f", exp(t[K+1]) );
    fprintf(fpV, "\nVvector <- c(%f", exp(t[K+2]) );

    fprintf(fpls0, "\nls0vector <- c(%f", logsigma[0]);
    fprintf(fpls1, "\nls1vector <- c(%f", logsigma[1]);
    fprintf(fpls2, "\nls2vector <- c(%f", logsigma[2]);
    fprintf(fplsmu, "\nlsmuvector <- c(%f", logsigma[K]);
    fprintf(fplsA, "\nlsAvector <- c(%f", logsigma[K+1]);
    fprintf(fplsV, "\nlsVvector <- c(%f", logsigma[K+2]);

    fprintf(fpe, "\nevector <- c(%f", pilogest);

    /* MAIN ITERATIVE LOOP. */
    for (batchnum=1; batchnum<=NUMBATCHES; batchnum++) {

	for (kk=0; kk<K+3; kk++) {
          acount[kk] = 0;
          if (abs(t[kk]-newt[kk]) > 0.01) {
            fprintf(stderr, "\nERROR: t and newt not synchronised.\n\n");
            exit(1);
          }
        }

        for (ii=1; ii<=BATCHLENGTH; ii++) {

	  numit++;
	  logval = log(1.0+sqnorm(t));

#ifdef EMPBAYES
	  for (kk=0; kk<K+2; kk++) {
#else
	  for (kk=0; kk<K+3; kk++) {
#endif

	    /* PROPOSE MOVE OF kk'th COORD. */
	    newt[kk] = t[kk] + exp(logsigma[kk]) * normal();
#ifdef SPEEDUP
	    if (kk<K)
	      logalpha = targlogdenscomp(newt, kk) - targlogdenscomp(t, kk);
	    else
#endif
	      logalpha = targlogdens(newt) - targlogdens(t);
#ifdef VERBOSE
  printf("Coord %d from %f to %f (ls=%f, la=%f) is ... ", 
			kk, t[kk], newt[kk], logsigma[kk], logalpha);
#endif
	    accepted = ( log(drand48()) < logalpha );
	    if (accepted) {
	      t[kk] = newt[kk];
	      acount[kk] = acount[kk] + 1;
#ifdef VERBOSE
  printf("accepted!\n");
#endif
	    } else {
	      newt[kk] = t[kk];
#ifdef VERBOSE
  printf("rejected!\n");
#endif
	    }

	  } /* End of kk for loop. */

	  logval = log(1.0+sqnorm(t));
	  pilogsum = pilogsum + logval;

	/* Write values to files. */
        fprintf(fpx0, ", %f", t[0]);
        fprintf(fpx1, ", %f", t[1]);
        fprintf(fpx2, ", %f", t[2]);
        fprintf(fpmu, ", %f", t[K]);
        fprintf(fpA, ", %f", exp(t[K+1]) );
        fprintf(fpV, ", %f", exp(t[K+2]) );

        fprintf(fpls0, ", %f", logsigma[0]);
        fprintf(fpls1, ", %f", logsigma[1]);
        fprintf(fpls2, ", %f", logsigma[2]);
        fprintf(fplsmu, ", %f", logsigma[K]);
        fprintf(fplsA, ", %f", logsigma[K+1]);
        fprintf(fplsV, ", %f", logsigma[K+2]);

	fprintf(fpe, ", %f", pilogest);

        } /* End of ii for loop. */

	/* Update various estimates etc. */
	pilogest = pilogsum / numit;

	/* DO THE ADAPTING. */

#ifdef ADAPTSIGMAS
	for (jj=0; jj<K+3; jj++) {
	    if (acount[jj] > BATCHLENGTH * TARGACCEPT )
	      logsigma[jj] = logsigma[jj] + adaptamount(batchnum);
	    else if (acount[jj] < BATCHLENGTH * TARGACCEPT )
	      logsigma[jj] = logsigma[jj] - adaptamount(batchnum);
	}
#endif

      if (divisible(batchnum, xspacing)) {
	/* Write values to files -- no, done earlier for now. */

      }

      if (divisible(batchnum, PRINTLENGTH)) {
        /* OUTPUT SOME VALUES TO SCREEN. */
        fflush(fpx0);
        fflush(NULL);
        for(jj=0; jj<K+3; jj++) {
          printf("t[%d]=%.2f, ls[%d]=%.2f; ", jj, t[jj], jj, logsigma[jj]);
        }
        printf("sq=%f, pilogest=%f ", sqnorm(t), pilogest );
        gettimeofday (&tmptv, (struct timezone *)NULL);
        currtime = 1.0 * tmptv.tv_sec + 0.000001 * tmptv.tv_usec;
        printf("(%d/%d, %.2f secs)\n", batchnum, NUMBATCHES,
						currtime-begintime);
     }

} /* End of batchnum for loop. */

fprintf(fpx0, " )\n\n");
fprintf(fpx1, " )\n\n");
fprintf(fpx2, " )\n\n");
fprintf(fpmu, " )\n\n");
fprintf(fpA, " )\n\n");
fprintf(fpV, " )\n\n");

fprintf(fpls0, " )\n\n");
fprintf(fpls1, " )\n\n");
fprintf(fpls2, " )\n\n");
fprintf(fplsmu, " )\n\n");
fprintf(fplsA, " )\n\n");
fprintf(fplsV, " )\n\n");

fprintf(fpe, " )\n\n");

fclose(fpx0);
fclose(fpx1);
fclose(fpx2);
fclose(fpmu);
fclose(fpA);
fclose(fpV);

fclose(fpls0);
fclose(fpls1);
fclose(fpls2);
fclose(fplsmu);
fclose(fplsA);
fclose(fplsV);

fclose(fpe);

fprintf(fpm, "\n\n");
fprintf(fpm, "source(\"%s\");\n", X0FILE);
fprintf(fpm, "source(\"%s\");\n", EFILE);
fprintf(fpm, "\n\n");

fprintf(fpm, "pdf(file=\"plotvaratx0.pdf\", width=4, height=4) \n plot( %d*(0:%d), x0vector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "pdf(file=\"plotvarate.pdf\", width=4, height=4) \n plot( %d*(0:%d), evector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);

fprintf(fpm, "\n");

fprintf(fpm, "postscript(file=\"plotvaratx0.eps\", width=4, height=4, horizontal = FALSE, onefile = FALSE, paper = \"special\") \n plot( %d*(0:%d), x0vector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "postscript(file=\"plotvarate.eps\", width=4, height=4, horizontal = FALSE, onefile = FALSE, paper = \"special\") \n plot( %d*(0:%d), evector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);

fprintf(fpm, "\n\n");
fclose(fpm);

gettimeofday (&tmptv, (struct timezone *)NULL);
endtime = 1.0 * tmptv.tv_sec + 0.000001 * tmptv.tv_usec;

printf("\nDone ... K = %d;  total number of variables = %d.\n", K, K+3);
printf("Batch size = %d full scans.\n", BATCHLENGTH);
printf("Printed to screen every %d batches.\n", PRINTLENGTH);
printf("Did %d batches in total.\n", NUMBATCHES);
for (jj=0; jj<K; jj++)
  printf("r[%d]=%d, ", jj, r[jj]);
printf("\n\n");
printf("Ellapsed time = %.2f seconds.\n\n", endtime-begintime);

return(0);

}


/* TARGLOGDENS */
double targlogdens( double t[] )
{
    double result, A, V, mu, targlogdenscomp();
    int kk, jj;

    mu = t[K];
    A = exp(t[K+1]);
    V = exp(t[K+2]);

    result = -(mu-mu0)*(mu-mu0) / 2.0 / s0 / s0
                    -b1/A - (a1+1) * log(A) -b2/V - (a2+1) * log(V)
                    + log(A) + log(V); /* to account for using log scale. */

/*
    for (kk=0; kk<K; kk++) {
	result = result - 0.5 * log(A) - (t[kk]-mu)*(t[kk]-mu) / 2.0 / A;
        for (jj=0; jj<r[kk]; jj++) {
          result = result - (Y[kk][jj]-t[kk])*(Y[kk][jj]-t[kk]) / 2.0 / V
            - 0.5 * log(V);
        }
    }
*/

    for (kk=0; kk<K; kk++) {
        result = result + targlogdenscomp(t, kk);
    }

    return(result);
}

double targlogdenscomp( double t[], int kk )
{
    double result, A, V, mu;
    double square();
    int jj;

    if ( (kk<0) || (kk>K-1) ) {
      fprintf(stderr, "\nERROR: component number out of range!\n\n");
      exit(1);
    }

    mu = t[K];
    A = exp(t[K+1]);
    V = exp(t[K+2]);

#ifdef CAUCHYLEVEL
	result = - log(A) - square( (t[kk]-mu) / A );
#else
	/* Usual normal case. */
	result = - 0.5 * log(A) - square(t[kk]-mu) / 2.0 / A;
#endif
        for (jj=0; jj<r[kk]; jj++) {
          result = result - square(Y[kk][jj]-t[kk]) / 2.0 / V
            - 0.5 * log(V);
        }

    return(result);
}


/* SQNORM */
double sqnorm( double t[] )
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


int divisible (int n1, int n2)
{
  return (n1 == n2 * (n1/n2));
}


double square(double xx)
{
    return(xx*xx);
}


