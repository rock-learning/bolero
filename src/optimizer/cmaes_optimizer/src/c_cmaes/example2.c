#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/times.h>
#include "cmaes_interface.h"

double f_rosenbrock( double *x);
double f_rand( double *x);
double f_constant( double *x);
double f_kugelmin1( double *x);
double f_sphere( double *x);
double f_stepsphere( double *x);
double f_cigar( double *x);
double f_cigtab( double *x);
double f_tablet( double *x);
double f_elli( double *x);
double f_elli100( double *x);
double f_ellinumtest( double *x);
double f_parabR( double *x);
double f_sharpR( double *x);
double f_diffpow( double *x);
double f_gleichsys5( double *x);
int optimize(double(*pFun)(double *), int dim, char *input_parameter_filename);
void writeoutput(cmaes_t *pevo);
/*___________________________________________________________________________
//___________________________________________________________________________
//
// smallest possible interface to optimize a function with cmaes_t,
// reads from file "incmaes.par" here and in cmaes_init()
//___________________________________________________________________________
*/ 
int main(int argn, char **args)
{
  struct tms before, after;
  typedef double (*pfun_t)(double *); 
  pfun_t rgpFun[99];  /* array (range) of pointer to objective function */
  char *filename;
  int nb;  /* objective function number to be used */ 
  int maxnb, dim, i;

  /* Put together objective functions */
  rgpFun[0] = f_sphere;        rgpFun[1] = f_elli; 
  rgpFun[2] = f_cigar;         rgpFun[3] = f_cigtab; 
  rgpFun[4] = f_tablet;        rgpFun[5] = f_rosenbrock; 
  rgpFun[6] = f_parabR;        rgpFun[7] = f_sharpR;
  rgpFun[8] = f_diffpow;       rgpFun[9] = f_kugelmin1;     
  rgpFun[10] = f_ellinumtest;  rgpFun[11] = f_elli100;      
  rgpFun[18] = f_gleichsys5;   rgpFun[19] = f_rand;         
  rgpFun[20] = f_constant;     rgpFun[21] = f_stepsphere;
  maxnb = 21; 


  /* set up everything */

  filename = "writeonly"; /* write parameters to file actparcmaes.par */
  filename = "non";       /* no parameter reading/writing at all */ 
  nb = 1;
  dim = 5;

  /* Optimize function */

  times(&before);

  for (i = 0; i < 100; ++i)
    optimize(rgpFun[nb], dim, filename);

  times(&after);
  
  printf("User time: %ld / 100 seconds\n", after.tms_utime -
						before.tms_utime);
  printf("System time: %ld / 100 seconds\n", after.tms_stime -
	 before.tms_stime);

  return 0;
} /* main() */

/*___________________________________________________________________________
//___________________________________________________________________________
//
// Somewhat extended interface for optimizing pFun with cmaes_t
//___________________________________________________________________________
*/ 

int optimize(double(*pFun)(double *), int dim, char * parameterkey)
{
  cmaes_t evo; 
  double *rgFunVal = NULL;
  double *const*rgx; /* pointer to sample population */
  int lambda, i;
  double *rgxstart, *rgsigma;

  /* set essential initialization parameters */

  rgxstart = calloc(dim, sizeof(double));
  rgsigma = calloc(dim, sizeof(double));
  for (i = 0; i < dim; ++i) {
    rgxstart[i] = 0.5;  // initial search point
    rgsigma[i] = 0.3;   // initial sigma
  }
  lambda = 0; /* 0,1 == invokes default */

  cmaes_init(&evo, pFun, dim, rgxstart, rgsigma, 0, lambda, parameterkey);

  /* more settings, quite a hack up to now */
  evo.sp.stStopFuncValue.flg = 1; 
  evo.sp.stStopFuncValue.val = 1e-12; // set function value to be achieved

  while(!cmaes_Test(&evo, "stop"))
    { 
      /* Generate sample population */
      rgx = cmaes_SampleDistribution(&evo, NULL); 

      /* Here optionally handle constraints etc. on rgx. You may call
      // cmaes_ReSampleSingle(rgx[i]) to resample the i-th sample. Do
      // not change rgx in any other way. You may also copy and
      // modify (repair) rgx[i] only for the evaluation of the fitness
      // function and consider adding a penalty depending on the size
      // of the modification.
      */

      /* The following can be replaced by any other assignment of
	 rgFunVal to a double array of (constant) length
	 len := cmaes_Get("lambda") == cmaes_Get("samplesize").
         rgFunVal[i], i=0...len-1 corresponds to the objective 
         function value of rgx[i], e.g. (*pFun)(rgx[i]). 
      */
      rgFunVal = cmaes_EvaluateSample(&evo, NULL); 

      /* reestimate search distribution */
      cmaes_ReestimateDistribution(&evo, rgFunVal); 

      /* read control signals and write "personal" output */
      // cmaes_ReadSignals(&evo, NULL); /* from file signals.par */

    } /* while !cmaes_Test(&evo, "stop") */
  // writeoutput(&evo);             /* write my own output, see below */
  cmaes_exit(&evo);
  return 0;
}

/*___________________________________________________________________________
//___________________________________________________________________________
//
// writing of output to files and console.
//___________________________________________________________________________
*/
void writeoutput(cmaes_t *pevo)
{
  char *recordingfile = "rescmaes.dat";         /* this is default anyhow */
  char *recordingxfile = "xcmaes.dat";
  char *recordingallfile = "allcmaes.dat"; 
  double outmod = cmaes_Get(pevo, "maxgen")/10; /* <=10 console outputs */
  double outmodwrite = 1e99; /* force writing any outmodwrite generation */
  static double outtime = 0; 
  static double lastgen = 0;
  static time_t lasttimeall = 0; 
  clock_t lasttime0; 
  double gen = cmaes_Get(pevo, "generation"); 

  /* last output */
  if (cmaes_Test(pevo, "stop")) 
    {
      int i;
      const double *x = cmaes_Getp(pevo, "x"); 
      printf("%.0f: xmean\n", cmaes_Get(pevo, "eval"));
      for (i=0; i<x[-1]; ++i)
	printf("  %+10.4e%c", x[i], (i%5==4||i==x[-1]-1) ? '\n' : ' ');

      printf("%.0f generations, %.0f fevals (%.1f sec): f(x)=%g\n", 
	     cmaes_Get(pevo, "gen"), cmaes_Get(pevo, "eval"), 
	     pevo->timings.totaltime,
	     cmaes_Get(pevo, "funval"));
      printf("  (axis-ratio=%.2e, max/min-stddev=%.2e/%.2e)\n", 
	     cmaes_Get(pevo, "maxmainax") / cmaes_Get(pevo, "minmainax"),
	     cmaes_Get(pevo, "maxstddev"), cmaes_Get(pevo, "minstddev") 
	     );
      printf("Stop: %s\n",  cmaes_Test(pevo, "stop"));
      cmaes_WriteToFile(pevo, "all", recordingallfile);
      cmaes_WriteToFile(pevo, "resume", recordingallfile);
    }
  /* print on console, this is obsolete, because 
     printing is possible via file signals.par */
  if (0 && (fmod(gen, outmod) < 1 || gen == 1  || 
	    cmaes_Test(pevo, "stop") != NULL)) {
    if (gen == 1)
      printf("%s%s", "   Gen.  Func.Value  Max.Std.Dev Min.Std.Dev ",
	     " Sigma   Axis-Ratio\n");
    printf("%7.0f %.6e %.5e %.5e %.2e %.4e \n", gen, 
	   cmaes_Get(pevo, "funval"), 
	   cmaes_Get(pevo, "maxstddev"), cmaes_Get(pevo, "minstddev"), 
	   pevo->sigma, 
	   cmaes_Get(pevo, "maxmainax") / cmaes_Get(pevo, "minmainax"));
  }
  /* write record of important parameters, more easily achieved by editing signals.par */
  if (0 && (fmod(gen, outmodwrite) < 1 
	    || (
		gen - lastgen > sqrt(gen) - 3  /* controls file size */
		&& 1 + 1e-3*gen + pevo->timings.totaltime 
		> 20 * outtime                 /* controls writing time */
		)
	    || cmaes_Test(pevo, "stop") != NULL)) {
    lasttime0 = clock();
    lastgen = gen; 
    cmaes_WriteToFile(pevo, "few+few(diag(D))", recordingfile);
    cmaes_WriteToFile(pevo, "eval+xmean", recordingxfile);
    outtime += (double)(clock()-lasttime0)/CLOCKS_PER_SEC;
  }
  /* write everything once per hour */
  if (difftime(time(NULL), lasttimeall) > 60*60) {
    cmaes_WriteToFile(pevo, "all", recordingallfile);
    lasttimeall = time(NULL);
  }
} /* writeoutput() */
#if 1  
/*___________________________________________________________________________
//___________________________________________________________________________
*/
double f_rand( double *x)
{
  return (double)rand() / RAND_MAX; 
}
double f_constant( double *x)
{
  return 1; 
}
#endif

static double SQR(double d)
{
  return (d*d);
}

/* ----------------------------------------------------------------------- */
double f_stepsphere( double *x)
{
  int i;
  double sum = 0.;
  int DIM = (int)(x[-1]);
  for (i = 0; i < DIM; ++i)
    sum += floor(x[i]*x[i]);
  return sum;
}

/* ----------------------------------------------------------------------- */
double f_sphere( double *x)
{
  int i;
  double sum = 0.;
  int DIM = (int)(x[-1]);
  for (i = 0; i < DIM; ++i)
    sum += x[i]*x[i];
  return sum;
}

/* ----------------------------------------------------------------------- */
double f_cigar( double *x)
{
  int i;
  double sum = 0.;
  int DIM = (int)(x[-1]);

  for (i = 1; i < DIM; ++i)
    sum += x[i]*x[i];
  sum *= 1e6;
  sum += x[0]*x[0];
  return sum;
}

/* ----------------------------------------------------------------------- */
double f_cigtab( double *x)
{
  int i;
  double sum = 0.;
  int DIM = (int)(x[-1]);

  sum = x[0]*x[0] + 1e8*x[DIM-1]*x[DIM-1];
  for (i = 1; i < DIM-1; ++i)
    sum += 1e4*x[i]*x[i];
  return sum;
}

/* ----------------------------------------------------------------------- */
double f_tablet( double *x)
{
  int i;
  double sum = 0.;
  int DIM = (int)(x[-1]);

  sum = 1e6*x[0]*x[0];
  for (i = 1; i < DIM; ++i)
    sum += x[i]*x[i];
  return sum;
}

/* ----------------------------------------------------------------------- */
double f_elli( double *x)
{
  int i;
  double sum = 0.;
  int DIM = (int)(x[-1]);
  
  if (DIM == 1)
    return x[0] * x[0];
  for (i = 0; i < DIM; ++i)
    sum += exp(log(1000) * 2. * (double)(i)/(DIM-1)) * x[i]*x[i];
  return sum;
}

/* ----------------------------------------------------------------------- */
double f_elli100( double *x)
{
  int i;
  double sum = 0.;
  int DIM = (int)(x[-1]);
  
  if (DIM == 1)
    return x[0] * x[0];
  for (i = 0; i < DIM; ++i)
    sum += exp(log(100) * 2. * (double)(i)/(DIM-1)) * x[i]*x[i];
  return sum;
}

/* ----------------------------------------------------------------------- */
double f_diffpow( double *x)
{
  int i;
  double sum = 0.;
  int DIM = (int)(x[-1]);
  
  if (DIM == 1)
    return x[0] * x[0];
  for (i = 0; i < DIM; ++i)
    sum += pow(fabs(x[i]), 2.+10*(double)(i)/(DIM-1));
  return sum;
}
/* ----------------------------------------------------------------------- */
double f_kugelmin1( double *x)
{
  int i;
  double sum = 0.;
  int DIM = (int)(x[-1]);

  for (i = 1; i < DIM; ++i)
    sum += x[i]*x[i];
  return sum;
}

/* ----------------------------------------------------------------------- */
double f_rosenbrock( double *x)
/*
	Rosenbrock's Function, generalized.
*/
{
  double qualitaet;
  int i;
  int DIM = (int)(x[-1]);
	qualitaet = 0.0;

	for( i = DIM-2; i >= 0; --i)
	  qualitaet += 100.*SQR(SQR(x[i])-x[i+1]) + SQR(1.-x[i]);
	return ( qualitaet);
} /* f_rosenbrock() */

/* ----------------------------------------------------------------------- */
double f_parabR( double *x)
{
  int i;
  double sum = 0.;
  int DIM = (int)(x[-1]);
  for (i = 1; i < DIM; ++i)
    sum += x[i]*x[i];
  return -x[0] + 100.*sum;
}

/* ----------------------------------------------------------------------- */
double f_sharpR( double *x)
{
  int i;
  double sum = 0.;
  int DIM = (int)(x[-1]);
  for (i = 1; i < DIM; ++i)
    sum += x[i]*x[i];
  return -x[0] + 100*sqrt(sum);
}

/* ----------------------------------------------------------------------- */
double f_ellinumtest( double *x)
{
  int i;
  double sum = 0.;
  int DIM = (int)(x[-1]);
  static double maxVerhaeltnis = 0.;
  if (maxVerhaeltnis == 0.)
    {
      for (maxVerhaeltnis = 1.; 
	   maxVerhaeltnis < 1e99 && maxVerhaeltnis < 2. * maxVerhaeltnis; 
	   maxVerhaeltnis *= 2.) 
	if (maxVerhaeltnis == maxVerhaeltnis + 1.)
	  break;
      maxVerhaeltnis *= 10.;
      maxVerhaeltnis = sqrt (maxVerhaeltnis);
    }
  if (DIM < 3)
    return x[0] * x[0];
  for (i = 1; i < DIM; ++i)
    sum += exp(log(maxVerhaeltnis) * 2. * (double)(i-1)/(DIM-2)) * x[i]*x[i];
  return sum;
}

/* ----------------------------------------------------------------------- */
double f_gleichsys5( double *x)
/*
	Gleichungssystem 5-dimensional von Juergen Bremer
	Fuer jede Zeile soll gelten:
	 c_1*x[1] + c_2*x[2] + c_3*x[3] + c_4*x[4] + c_5*x[5] + c_0 = 0 
	 Deswegen geht das Quadrat der linken Seite in die 
	 Qualitaetsfunktion ein. 
*/
{
  double qualitaet = 0.0;

#if 1
  static double koeff[5][6] =
    {/* c_1,   c_2,  c_3,   c_4,  c_5,   c_0 */
      {   4,   191,   27,   199,   21,   172},
      { 191, 10883, 1413,  5402,  684, -8622},
      {  27,  1413,  191,  1032,  118,   -94}, 
      { 199,  5402, 1032, 29203, 2331, 78172}, 
      {  21,   684,  118,  2331,  199,  5648}
    };
  int i, j;
  double sum; 
 
  for( i = 0; i < 5; ++i)
    {
      sum = koeff[i][5];
      for ( j = 0; j < 5; ++j)
	{
	  sum += koeff[i][j] * x[j];
	}
      qualitaet += sum * sum;
    }
#endif
  return qualitaet;
} /* f_gleichsys5() */


/*
  05/10/05: revised buggy comment on handling constraints by resampling 
*/
