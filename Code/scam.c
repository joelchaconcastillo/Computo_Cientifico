
/* 

==========================================================================

scam.c -- a program to run the Haario et al. "SCAM" algorithm

Copyright (c) 2007 by Jeffrey S. Rosenthal (jeff@math.toronto.edu).

Licensed for general copying, distribution and modification according to
the GNU General Public License (http://www.gnu.org/copyleft/gpl.html).

----------------------------------------------------

Compile with "cc scam.c -lm", then run with "a.out".

==========================================================================

*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* #define NUMITER 500000 */
#define NUMITER 500000

/* #define EMPBAYES true */

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

#define PRINTLENGTH 100

#define TARGACCEPT 0.44
/* #define TARGACCEPT 0.234 */

#define PI 3.14159265

#define X0FILE "scamx0"
#define X1FILE "scamx1"
#define X2FILE "scamx2"
#define MUFILE "scammu"
#define AFILE "scamA"
#define VFILE "scamV"
#define SG0FILE "scamsg0"
#define SG1FILE "scamsg1"
#define SG2FILE "scamsg2"
#define SGMUFILE "scamsgmu"
#define SGAFILE "scamsgA"
#define SGVFILE "scamsgV"
#define EFILE "scame"
#define MASTERFILE "scamm"

double drand48();

int r[K];
double mm[K], sd[K];

#define JMAX 1000

double Y[K][JMAX];

main()

{

    int i,j,k,ii, jj, kk, xspacing;
    int itnum, accepted, acount[K+3];
    double t[K+3];
    double newt[K+3];
    double sigma[K+3], xbar[K+3], g[K+3], newxbar, newg;
    double sqdiff, logalpha, tmp;
    double pilogest, pilogsum, logval;
    double sqnorm(), normal(), targlogdens(), targlogdenscomp(), adaptamount();
    double square();
    FILE *fpx0, *fpx1, *fpx2, *fpmu, *fpA, *fpV, *fpe, *fpm;
    FILE *fpsg0, *fpsg1, *fpsg2, *fpsgmu, *fpsgA, *fpsgV;
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
    itnum = 0;
    /* mu = 0.0; */
    /* A = V = 1.0; */
    for (i=0; i<K+3; i++) {
	t[i] = xbar[i] = g[i] = 0.0;
	sigma[i] = 5;
    }
#ifdef EMPBAYES
    t[K+2] = 0.00434;
#endif
    logval = pilogest = pilogsum = 0.0;
    for (kk=0; kk<K+3; kk++)
	newt[kk] = t[kk];

    gettimeofday (&tmptv, (struct timezone *)NULL);
    begintime = 1.0 * tmptv.tv_sec + 0.000001 * tmptv.tv_usec;

    printf("\nBeginning variable-at-time adaption run.\n");
    printf("K = %d;  total number of variables = %d.\n", K, K+3);
    printf("Printing to screen every %d iterations.\n", PRINTLENGTH);
    printf("Doing %d full scans in total.\n", NUMITER);
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

    if ((fpsg0 = fopen(SG0FILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", SG0FILE);
    }
    if ((fpsg1 = fopen(SG1FILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", SG1FILE);
    }
    if ((fpsg2 = fopen(SG2FILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", SG2FILE);
    }
    if ((fpsgmu = fopen(SGMUFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", SGMUFILE);
    }
    if ((fpsgA = fopen(SGAFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", SGAFILE);
    }
    if ((fpsgV = fopen(SGVFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", SGVFILE);
    }

    if ((fpe = fopen(EFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", EFILE);
    }
    if ((fpm = fopen(MASTERFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", MASTERFILE);
    }

    xspacing = NUMITER / 10000;
    if (xspacing == 0)
        xspacing = 1;

    fprintf(fpx0, "\nx0vector <- c(%f", t[0]);
    fprintf(fpx1, "\nx1vector <- c(%f", t[1]);
    fprintf(fpx2, "\nx2vector <- c(%f", t[2]);
    fprintf(fpmu, "\nmuvector <- c(%f", t[K]);
    fprintf(fpA, "\nAvector <- c(%f", exp(t[K+1]) );
    fprintf(fpV, "\nVvector <- c(%f", exp(t[K+2]) );

    fprintf(fpsg0, "\nsg0vector <- c(%f", sigma[0]);
    fprintf(fpsg1, "\nsg1vector <- c(%f", sigma[1]);
    fprintf(fpsg2, "\nsg2vector <- c(%f", sigma[2]);
    fprintf(fpsgmu, "\nsgmuvector <- c(%f", sigma[K]);
    fprintf(fpsgA, "\nsgAvector <- c(%f", sigma[K+1]);
    fprintf(fpsgV, "\nsgVvector <- c(%f", sigma[K+2]);

    fprintf(fpe, "\nevector <- c(%f", pilogest);

    /* MAIN ITERATIVE LOOP. */
    for (itnum=1; itnum<=NUMITER; itnum++) {

	for (kk=0; kk<K+3; kk++) {
          acount[kk] = 0;
          if (abs(t[kk]-newt[kk]) > 0.01) {
            fprintf(stderr, "\nERROR: t and newt not synchronised.\n\n");
            exit(1);
          }
        }

	  logval = log(1.0+sqnorm(t));

#ifdef EMPBAYES
	  for (kk=0; kk<K+2; kk++) {
#else
	  for (kk=0; kk<K+3; kk++) {
#endif

	    /* UPDATE VALUES FOR kk'th COORD. */
	    tmp = (double) itnum;
	    newxbar = ((tmp-1)/itnum) * xbar[kk] + t[kk] / tmp;
	    if (itnum <= 1)
	      newg = 0.0;
	    else
	      newg = (tmp-2)/(tmp-1) * g[kk] + square(xbar[kk]) +
		square(t[kk])/(tmp-1) - (tmp/(tmp-1)) * square(newxbar);
	    if (itnum <= 10) {
	      sigma[kk] = 5;
	    } else {
	      sigma[kk] = 2.4 * sqrt( newg + 0.05 );
	    }
	    xbar[kk] = newxbar;
	    g[kk] = newg;

	    /* PROPOSE MOVE OF kk'th COORD. */
	    newt[kk] = t[kk] + sigma[kk] * normal();
#ifdef SPEEDUP
	    if (kk<K)
	      logalpha = targlogdenscomp(newt, kk) - targlogdenscomp(t, kk);
	    else
#endif
	      logalpha = targlogdens(newt) - targlogdens(t);
#ifdef VERBOSE
  printf("Coord %d from %f to %f (sg=%f, la=%f) is ... ", 
			kk, t[kk], newt[kk], sigma[kk], logalpha);
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

        fprintf(fpsg0, ", %f", sigma[0]);
        fprintf(fpsg1, ", %f", sigma[1]);
        fprintf(fpsg2, ", %f", sigma[2]);
        fprintf(fpsgmu, ", %f", sigma[K]);
        fprintf(fpsgA, ", %f", sigma[K+1]);
        fprintf(fpsgV, ", %f", sigma[K+2]);

	fprintf(fpe, ", %f", pilogest);

	/* Update various estimates etc. */
	pilogest = pilogsum / itnum;

      if (divisible(itnum, xspacing)) {
	/* Write values to files -- no, done earlier for now. */

      }

      if (divisible(itnum, PRINTLENGTH)) {
        /* OUTPUT SOME VALUES TO SCREEN. */
        fflush(fpx0);
        fflush(NULL);
        for(jj=0; jj<K+3; jj++) {
          printf("t[%d]=%.2f, sg[%d]=%.2f; ", jj, t[jj], jj, sigma[jj]);
        }
        printf("sq=%f, pilogest=%f ", sqnorm(t), pilogest );
        gettimeofday (&tmptv, (struct timezone *)NULL);
        currtime = 1.0 * tmptv.tv_sec + 0.000001 * tmptv.tv_usec;
        printf("(%d/%d, %.2f secs)\n", itnum, NUMITER, currtime-begintime);
      }

} /* End of itnum for loop. */

fprintf(fpx0, " )\n\n");
fprintf(fpx1, " )\n\n");
fprintf(fpx2, " )\n\n");
fprintf(fpmu, " )\n\n");
fprintf(fpA, " )\n\n");
fprintf(fpV, " )\n\n");

fprintf(fpsg0, " )\n\n");
fprintf(fpsg1, " )\n\n");
fprintf(fpsg2, " )\n\n");
fprintf(fpsgmu, " )\n\n");
fprintf(fpsgA, " )\n\n");
fprintf(fpsgV, " )\n\n");

fprintf(fpe, " )\n\n");

fclose(fpx0);
fclose(fpx1);
fclose(fpx2);
fclose(fpmu);
fclose(fpA);
fclose(fpV);

fclose(fpsg0);
fclose(fpsg1);
fclose(fpsg2);
fclose(fpsgmu);
fclose(fpsgA);
fclose(fpsgV);

fclose(fpe);

fprintf(fpm, "\n\n");
fprintf(fpm, "source(\"%s\");\n", X0FILE);
fprintf(fpm, "source(\"%s\");\n", EFILE);
fprintf(fpm, "\n\n");

fprintf(fpm, "pdf(file=\"plotvaratx0.pdf\", width=4, height=4) \n plot( %d*(0:%d), x0vector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMITER/xspacing);
fprintf(fpm, "pdf(file=\"plotvarate.pdf\", width=4, height=4) \n plot( %d*(0:%d), evector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMITER/xspacing);

fprintf(fpm, "\n");

fprintf(fpm, "postscript(file=\"plotvaratx0.eps\", width=4, height=4, horizontal = FALSE, onefile = FALSE, paper = \"special\") \n plot( %d*(0:%d), x0vector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMITER/xspacing);
fprintf(fpm, "postscript(file=\"plotvarate.eps\", width=4, height=4, horizontal = FALSE, onefile = FALSE, paper = \"special\") \n plot( %d*(0:%d), evector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMITER/xspacing);

fprintf(fpm, "\n\n");
fclose(fpm);

gettimeofday (&tmptv, (struct timezone *)NULL);
endtime = 1.0 * tmptv.tv_sec + 0.000001 * tmptv.tv_usec;

printf("\nDone ... K = %d;  total number of variables = %d.\n", K, K+3);
printf("Printed to screen every %d batches.\n", PRINTLENGTH);
printf("Did %d full-scan iterations in total.\n", NUMITER);
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


