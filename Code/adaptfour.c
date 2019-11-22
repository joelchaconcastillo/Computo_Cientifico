
/* 

==========================================================================

ADAPTFOUR.C -- a program for doing multi-dim adaptive MCMC

Copyright (c) 2006 by Jeffrey S. Rosenthal (jeff@math.toronto.edu).

Licensed for general copying, distribution and modification according to
the GNU General Public License (http://www.gnu.org/copyleft/gpl.html).

----------------------------------------------------

Save as "adaptfour.c".

Compile with "cc adaptfour.c -lm", then run with "a.out".

Upon completion, can run 'source("adaptfourm")' in R to create plots.

Should find that mean of t[0] is about 0.313, and that
	mean of logval (evector) is about 0.0945 ...

==========================================================================

*/


#include <stdio.h>
#include <math.h>
#include <sys/time.h>

/* #define NUMBATCHES 100000 */
#define NUMBATCHES 100000

#define EMPBAYES true

#define BATCHLENGTH 100

/* #define DIMENSION 10 */

/* #define VERBOSE */

/* #define OTHERVERBOSE */

#define K 18

#define a1 -1
#define a2 -1
#define b1 2
#define b2 2
#define mu0 0
#define s0 1

#define PRINTLENGTH 1

/* #define TARGACCEPT 0.45 */
#define TARGACCEPT 0.234

#define PI 3.14159265

/* #define CUTOFF 3.085 */
/* #define CUTOFF 0.07 */
/* #define CUTOFF 10000.0 */
/* #define CUTOFF 0.09 */
/* #define CUTOFF 0.1444 */
/* #define CUTOFF 0.1 */
#define CUTOFF 1.6

#define INFINITY 999999999.0

#define XFILE "adaptfourx"
#define AFILE "adaptfoura"
#define BFILE "adaptfourb"
#define EFILE "adaptfoure"
#define MASTERFILE "adaptfourm"

/* #define CAUCHYTARG true */

#define MAXABSA 100.0
#define MAXABSB 100.0

double drand48();

double Y[K];


main()

{

    int i,j,k,batchnum, ii, jj, printcount, xspacing;
    int numit, numaccept, accepted;
    int faraccept, nearaccept, farsum, nearsum;
    double A, V, mu, t[K];
    double newA, newV, newmu, newt[K];
    double sigma, a, b;
    double sqdiff, logalpha;
    double pilogest, pilogsum, logval;
    double sqnorm(), normal(), targlogdens(), adaptamount();
    FILE *fpx, *fpa, *fpb, *fpe, *fpm;

    /* DATA VALUES */
    Y[0] = 0.395;
    Y[1] = 0.375;
    Y[2] = 0.355;
    Y[3] = 0.334;
    Y[4] = 0.313;
    Y[5] = 0.313;
    Y[6] = 0.291;
    Y[7] = 0.269;
    Y[8] = 0.247;
    Y[9] = 0.247;
    Y[10] = 0.224;
    Y[11] = 0.224;
    Y[12] = 0.224;
    Y[13] = 0.224;
    Y[14] = 0.224;
    Y[15] = 0.200;
    Y[16] = 0.175;
    Y[17] = 0.148;

    /* INITIALISATIONS. */
    seedrand();
    numit = 0;
    A = V = 1.0;
#ifdef EMPBAYES
    V = 0.00434;
#endif
    mu = 0.0;
    for (i=0; i<K; i++)
	t[i] = 0.0;
    sigma = 1.0;
    a = b = 0.0;
    /* a = -6.0;
    b = -6.0; */
    logval = pilogest = pilogsum = 0.0;
    printf("\nBeginning \"pi(1+log|x|)\" adaption run.\n");
    printf("\nAdapting every %d iterations.\n", BATCHLENGTH);
    printf("\nPrinting every %d iterations.\n", BATCHLENGTH*PRINTLENGTH);
    printf("\n");
    if ((fpx = fopen(XFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", XFILE);
    }
    if ((fpa = fopen(AFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", AFILE);
    }
    if ((fpb = fopen(BFILE,"w")) == NULL) {
        fprintf(stderr, "Unable to write to file %s.\n", BFILE);
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

    fprintf(fpx, "\nxvector <- c(%f", t[0]);
    fprintf(fpa, "\navector <- c(%f", a);
    fprintf(fpb, "\nbvector <- c(%f", b);
    fprintf(fpe, "\nevector <- c(%f", pilogest);

    /* MAIN ITERATIVE LOOP. */
    for (batchnum=1; batchnum<=NUMBATCHES; batchnum++) {

      for (printcount=0; printcount<PRINTLENGTH; printcount++) {

	/* Zero some counters. */
	numaccept = farsum = nearsum = faraccept = nearaccept = 0;

        for (ii=1; ii<=BATCHLENGTH; ii++) {

	  logval = log(1.0+sqnorm(A,V,mu,t));

	  /* GENERATE PROPOSAL VALUE. */
	  if (sqnorm(A,V,mu,t) < CUTOFF)
	    sigma = exp( a );
	  else
	    sigma = exp( b );
	  newA = A + sigma * normal();
	  newV = V + sigma * normal();
#ifdef EMPBAYES
	  newV = 0.00434;
#endif
	  newmu = mu + sigma * normal();
	  for (jj=0; jj<K; jj++)
	    newt[jj] = t[jj] + sigma * normal();

	  /* ACCEPT/REJECT. */
	  /* acc prob = \pi (y) / \pi (x)  * (\sigma_x / \sigma_y)^DIM *
                      exp( -(x-y)^2/2 (sigma_y^{-2} - \sigma_x^{-2}) ) */
	  logalpha = targlogdens(newA,newV,newmu,newt) - targlogdens(A,V,mu,t);
	  sqdiff = (newA-A)*(newA-A) + (newV-V)*(newV-V) +
	  					(newmu-mu)*(newmu-mu);
	  for (jj=0; jj<K; jj++)
	    sqdiff = sqdiff + (newt[jj]-t[jj])*(newt[jj]-t[jj]);
	  if ( (sqnorm(A,V,mu,t) < CUTOFF)
			&& (sqnorm(newA,newV,newmu,newt) >= CUTOFF) ) {
	    logalpha = logalpha + (a - b) * (K+3)
		- sqdiff/2.0 * (exp(-2*b) - exp(-2*a));
	  } else if ( (sqnorm(A,V,mu,t) >= CUTOFF)
			&& (sqnorm(newA,newV,newmu,newt) < CUTOFF) ) {
	    logalpha = logalpha + (b - a) * (K+3)
		- sqdiff/2.0 * (exp(-2*a) - exp(-2*b));
	  }
	  accepted = ( log(drand48()) < logalpha );

#ifdef VERBOSE
printf("oldsqnorm=%f, oldlog=%f, newlog=%f, logalpha=%f, acc=%d\n",
  sqnorm(A,V,mu,t), targlogdens(A,V,mu,t), targlogdens(newA,newV,newmu,newt),
	  	logalpha, accepted);
#endif

#ifdef OTHERVERBOSE
printf("A=%f, V=%f, t[0]=%f, t[1]=%f\n", A,V,t[0],t[1]);
#endif

	  /* Update the near and far counts. */
	  if (sqnorm(A,V,mu,t) < CUTOFF) {
	    nearsum++;
	    if (accepted) {
	      nearaccept++;
	    }
	  } else {
	    farsum++;
	    if (accepted) {
	      faraccept++;
	    }
	  }

	  /* Move if accepted. */
	  if (accepted) {
	      A = newA;
	      V = newV;
	      mu = newmu;
	      for (jj=0; jj<K; jj++)
		  t[jj] = newt[jj];
	      numaccept++;
	  }

	  /* Update various counts. */
	  numit++;

	  logval = log(1.0+sqnorm(A,V,mu,t));
	  pilogsum = pilogsum + logval;

        } /* End of ii for loop. */

	/* Update various estimates etc. */
	pilogest = pilogsum / numit;

	/* DO THE ADAPTING. */

/* printf("Adaption parameters: nearaccept=%d, nearsum=%d, faraccept=%d, * farsum=%d\n", nearaccept, nearsum, faraccept, farsum); */

	/* Adapt a. */
	if (nearaccept > nearsum * TARGACCEPT ) {
	  a = a + adaptamount(batchnum);
	} else if (nearaccept < nearsum * TARGACCEPT ) {
	  a = a - adaptamount(batchnum);
	}

	/* Adapt b. */
	if (faraccept > farsum * TARGACCEPT ) {
	  b = b + adaptamount(batchnum);
	} else if (faraccept < farsum * TARGACCEPT ) {
	  b = b - adaptamount(batchnum);
	}

	/* Prevent a from getting too extreme. */
	if (a > MAXABSA)
	  a = MAXABSA;
	if (a < -MAXABSA)
	  a = -MAXABSA;

	/* Prevent b from getting too extreme. */
	if (b > MAXABSB)
	  b = MAXABSB;
	if (b < -MAXABSB)
	  b = -MAXABSB;

      } /* End of printcount for loop. */

      /* OUTPUT SOME VALUES TO SCREEN. */
      /* printf("%9d: x[0]=%.3f, acc=%.3f, pilogest=%f, a:=%.5f, b:=%.5f\n", 
	   numit, x[0], ((double)numaccept)/BATCHLENGTH, pilogest, a, b); */
      printf("%9d: nearacc=%.3f, faracc=%f, a:=%.5f, b:=%.5f, sq=%f\n",
	   			numit, ((double)nearaccept)/nearsum,
				((double)faraccept)/farsum, a, b,
				sqnorm(A,V,mu,t) );

      if (batchnum == xspacing * (batchnum/xspacing)) {
	/* Write values to files. */
	fprintf(fpa, ", %f", a);
	fprintf(fpb, ", %f", b);
	fprintf(fpe, ", %f", pilogest);
	fprintf(fpx, ", %f", t[0]);
/* printf("printing!!\n"); */
      }

} /* End of batchnum for loop. */

fprintf(fpa, " )\n\n");
fprintf(fpb, " )\n\n");
fprintf(fpe, " )\n\n");
fprintf(fpx, " )\n\n");

fclose(fpx);
fclose(fpa);
fclose(fpb);
fclose(fpe);

fprintf(fpm, "\n\n");
fprintf(fpm, "source(\"%s\");\n", XFILE);
fprintf(fpm, "source(\"%s\");\n", AFILE);
fprintf(fpm, "source(\"%s\");\n", BFILE);
fprintf(fpm, "source(\"%s\");\n", EFILE);
fprintf(fpm, "\n\n");

fprintf(fpm, "pdf(file=\"plotfourx.pdf\", width=4, height=4) \n plot( %d*(0:%d), xvector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "pdf(file=\"plotfoura.pdf\", width=4, height=4) \n plot( %d*(0:%d), avector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "pdf(file=\"plotfourb.pdf\", width=4, height=4) \n plot( %d*(0:%d), bvector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "pdf(file=\"plotfoure.pdf\", width=4, height=4) \n plot( %d*(0:%d), evector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);

fprintf(fpm, "\n");

fprintf(fpm, "postscript(file=\"plotfourx.eps\", width=4, height=4, horizontal = FALSE, onefile = FALSE, paper = \"special\") \n plot( %d*(0:%d), xvector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "postscript(file=\"plotfoura.eps\", width=4, height=4, horizontal = FALSE, onefile = FALSE, paper = \"special\") \n plot( %d*(0:%d), avector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "postscript(file=\"plotfourb.eps\", width=4, height=4, horizontal = FALSE, onefile = FALSE, paper = \"special\") \n plot( %d*(0:%d), bvector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "postscript(file=\"plotfoure.eps\", width=4, height=4, horizontal = FALSE, onefile = FALSE, paper = \"special\") \n plot( %d*(0:%d), evector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);

fprintf(fpm, "\n\n");

fclose(fpm);
return(0);

}


/* TARGLOGDENS */
double targlogdens( double A, double V, double mu, double t[] )
{
    double result;
    int kk;

    if ( (A<0) || (V<0) )
	return(-INFINITY);

    result = -(mu-mu0)*(mu-mu0) / 2.0 / s0 / s0
                    -b1/A - (a1+1) * log(A) -b2/V - (a2+1) * log(V);

    for (kk=0; kk<K; kk++) {
	result = result - 0.5 * log(A) - (t[kk]-mu)*(t[kk]-mu) / 2.0 / A
            - 0.5 * log(V) - (Y[kk]-t[kk])*(Y[kk]-t[kk]) / 2.0 / V;
    }

    return(result);
}

/* SQNORM */
double sqnorm( double A, double V, double mu, double t[] )
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


