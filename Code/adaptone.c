
/* 

==========================================================================

ADAPTONE.C -- a program for doing one-dim adaptive MCMC

Copyright (c) 2006 by Jeffrey S. Rosenthal (jeff@math.toronto.edu).

Licensed for general copying, distribution and modification according to
the GNU General Public License (http://www.gnu.org/copyleft/gpl.html).

----------------------------------------------------

Save as "adaptone.c".

Compile with "cc adaptone.c -lm", then run with "a.out".

Upon completion, can run 'source("adaptonem")' in R to create plots.

Normal case: Can check pilogest in Mathematica (0.534822) with command:
NIntegrate[ (2*Pi)^(-0.5) * E^(-x*x/2) * Log[1+Abs[x]], {x,-Infinity,Infinity}]
Cauchy case: Can check pilogest in Mathematica (0.929695) with command:
NIntegrate[ 1/(1+x*x) * Log[1+Abs[x]], {x,-Infinity,Infinity}] / NIntegrate[ 1/(1+x*x),  {x,-Infinity,Infinity}]

==========================================================================

*/


#include <stdio.h>
#include <math.h>
#include <sys/time.h>

/* #define NUMBATCHES 100000 */
#define NUMBATCHES 100000

#define BATCHLENGTH 100

#define PRINTLENGTH 1

/* #define TARGACCEPT 0.234 */
/* #define TARGACCEPT 0.45 */
#define TARGACCEPT 0.44

#define PI 3.14159265

#define XFILE "adaptonex"
#define AFILE "adaptonea"
#define BFILE "adaptoneb"
#define EFILE "adaptonee"
#define MASTERFILE "adaptonem"

/* #define CAUCHYTARG true */

#define MAXABSA 100.0
#define MAXABSB 100.0

double drand48();

main()

{

    int i,j,k,batchnum, ii, printcount, xspacing;
    int numit, numaccept, accepted;
    int posaccept, negaccept;
    double x, logsigma, sigma, a, b, y, A, logalpha;
    double possum, negsum, pilogest, pilogsum, logval;
    double absval(), normal(), targlogdens(), adaptamount();
    FILE *fpx, *fpa, *fpb, *fpe, *fpm;

    /* INITIALISATIONS. */
    seedrand();
    numit = 0;
    x = 0.0;
    sigma = 1.0;
    a = 0.0;
    b = 1.0;
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

    fprintf(fpx, "\nxvector <- c(%f, ", x);
    fprintf(fpa, "\navector <- c(%f, ", a);
    fprintf(fpb, "\nbvector <- c(%f, ", b);
    fprintf(fpe, "\nevector <- c(%f, ", pilogest);

    /* MAIN ITERATIVE LOOP. */
    for (batchnum=1; batchnum<=NUMBATCHES; batchnum++) {

      for (printcount=0; printcount<PRINTLENGTH; printcount++) {

	/* Zero some counters. */
	numaccept = possum = negsum = posaccept = negaccept = 0;

        for (ii=1; ii<=BATCHLENGTH; ii++) {

	  /* GENERATE PROPOSAL VALUE. */
	  sigma = exp( a/2 + b/2*( logval - pilogest ) );
	  /* sigma = exp( a/2 + b/2*logval ); */
	  y = x + sigma * normal();
/* printf("CHECKING: x=%f, sigma=%f, y=%f\n", x, sigma, y); */

	  /* ACCEPT/REJECT. */
/* acceptance prob = \pi (y) / \pi (x)  * \sigma_x / \sigma _y * 
                   exp( -(x-y)^2/2 (sigma_y^{-2} - \sigma_x^{-2}) ) */
	  logalpha = targlogdens(y) - targlogdens(x)
                        + b/2 * ( log(1.0+absval(x)) - log(1.0+absval(y)) )
                        - (x-y)*(x-y)/2 * 
	    ( ( exp( -a-b*(log(1.0+absval(y))-pilogest) )
	    - exp( -a-b*(log(1.0+absval(x))-pilogest) ) ) );
			    /* ( ( exp( -a-b*log(1.0+absval(y)) )
			    - exp( -a-b*log(1.0+absval(x)) ) ) ); */
/* printf("targlogdens(y) = %f;  targlogdens(x) = %f;  logalpha = %f\n",
     targlogdens(y), targlogdens(x), logalpha); */
	  accepted = ( log(drand48()) < logalpha );
/* printf("x=%f, y=%f, ratio=%f, alpha=%f, accepted=%d, tst=%f\n", x,y,
	exp(targlogdens(y) - targlogdens(x)), alpha, accepted,
		b/2 * ( log(1.0+absval(x)) - log(1.0+absval(y)) ) ); */
	  if (accepted) {
	      x = y;
	      numaccept++;
	  }

	  /* Update various counts. */
	  numit++;

/* if (x<-1) m1count++; */
/* printf("TESTER: numit=%d, m1count=%d\n", numit, m1count); */

	  logval = log(1.0+absval(x));
	  pilogsum = pilogsum + logval;

	  if (logval > pilogest) {
	    possum++;
	    if (accepted) {
	      posaccept++;
	    }
	  } else {
	    negsum++;
	    if (accepted) {
	      negaccept++;
	    }
	  }

        } /* End of ii for loop. */

	/* Update various estimates etc. */
	pilogest = pilogsum / numit;

	/* DO THE ADAPTING. */

	/* Adapt a. */
	if (numaccept > BATCHLENGTH * TARGACCEPT ) {
	  a = a + adaptamount(batchnum);
	} else {
	  a = a - adaptamount(batchnum);
	}

/*
> If average acceptance probability in the region where
>
> \log (1 + |x| ) - E_\pi (\log (1+|x|)) > 0
>
> is smaller than that in the region's compelment then decrease b by 1/i
> otherwise increase it by 1/i.
*/

	/* Adapt b. */
	/* if ( posaccept * negcount < negaccept * poscount ) { */
	if ( posaccept * negsum < negaccept * possum ) {
	  b = b - adaptamount(batchnum);
	} else {
	  b = b + adaptamount(batchnum);
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
      printf("%9d: x=%.3f, acc=%.3f, pilogest=%f, a:=%.5f, b:=%.5f\n", 
		numit, x, ((double)numaccept)/BATCHLENGTH, pilogest, a, b);

      if (batchnum == xspacing * (batchnum/xspacing)) {
	/* Write values to files. */
	if (batchnum > xspacing) { /* Not first one printed, so need commas. */
	  fprintf(fpa, ",");
	  fprintf(fpb, ",");
	  fprintf(fpe, ",");
	  fprintf(fpx, ",");
	}
	fprintf(fpa, " %f", a);
	fprintf(fpb, " %f", b);
	fprintf(fpe, " %f", pilogest);
	fprintf(fpx, ", %f", x);
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

fprintf(fpm, "pdf(file=\"plotonex.pdf\", width=4, height=4) \n plot( %d*(0:%d), xvector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "pdf(file=\"plotonea.pdf\", width=4, height=4) \n plot( %d*(0:%d), avector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "pdf(file=\"plotoneb.pdf\", width=4, height=4) \n plot( %d*(0:%d), bvector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "pdf(file=\"plotonee.pdf\", width=4, height=4) \n plot( %d*(0:%d), evector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);

fprintf(fpm, "\n");

fprintf(fpm, "postscript(file=\"plotonex.eps\", width=4, height=4, horizontal = FALSE, onefile = FALSE, paper = \"special\") \n plot( %d*(0:%d), xvector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "postscript(file=\"plotonea.eps\", width=4, height=4, horizontal = FALSE, onefile = FALSE, paper = \"special\") \n plot( %d*(0:%d), avector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "postscript(file=\"plotoneb.eps\", width=4, height=4, horizontal = FALSE, onefile = FALSE, paper = \"special\") \n plot( %d*(0:%d), bvector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);
fprintf(fpm, "postscript(file=\"plotonee.eps\", width=4, height=4, horizontal = FALSE, onefile = FALSE, paper = \"special\") \n plot( %d*(0:%d), evector, type='l', xlab=\"\", ylab=\"\") \n dev.off()\n\n", xspacing, NUMBATCHES/xspacing);

fprintf(fpm, "\n\n");

fclose(fpm);
return(0);

}


/* TARGLOGDENS */
double targlogdens( double w )
{
#ifdef CAUCHYTARG
    return( -log(1.0+w*w) );
#endif
    return(-w*w/2);
}

/* ABSVAL */
double absval( double w )
{
  if (w<0)
    return(-w);
  else
    return(w);
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
    /* return( 100.0 / imax(10000, sqrt(iteration)) ); */
    return(0.01);
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


