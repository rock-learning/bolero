/* --------------------------------------------------------- */
/* --- File: cmaes_interface.h - Author: Nikolaus Hansen --- */
/* ---------------------- last modified: II 2006         --- */
/* --------------------------------- by: Nikolaus Hansen --- */
/* --------------------------------------------------------- */
/*   
     CMA-ES for non-linear function minimization. 

     Copyright (C) 1996, 2003  Nikolaus Hansen. 
     e-mail: hansen@bionik.tu-berlin.de

     This library is free software; you can redistribute it and/or
     modify it under the terms of the GNU Lesser General Public
     License as published by the Free Software Foundation; either
     version 2.1 of the License, or (at your option) any later 
     version (see http://www.gnu.org/copyleft/lesser.html).
     
     This library is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
     Lesser General Public License for more details.

*/
#ifndef NH_cmaes_interface_h /* only include once */ 
#define NH_cmaes_interface_h 
#include "cmaes.h"
/* --------------------------------------------------------- */
/* ------------------ Interface ---------------------------- */
/* --------------------------------------------------------- */
void cmaes_init(cmaes_t *cma_struct_ptr, double(*pFun)(double *), 
                int dimension , double *xstart, double *sigma,      
                long seed, int lambda, const char *input_parameter_filename);
void cmaes_resume_distribution(cmaes_t *cmaes_struct_ptr, char *filename);
void cmaes_exit(cmaes_t *);

double * const * cmaes_SampleDistribution(cmaes_t *evo_ptr, 
                                          const double *xmean);
double *         cmaes_EvaluateSample(cmaes_t *,
                                      double(*pFun)(double *));
double *         cmaes_ReestimateDistribution(cmaes_t *, 
                                              const double *rgFuncValue);

double         cmaes_Get(cmaes_t *, char const *szKeyWord);
const double * cmaes_Getp(cmaes_t *, char const *szName);/* e.g. "xbestever" */
void           cmaes_ReadSignals(cmaes_t *t, char *filename);
double *       cmaes_ReSampleSingle(cmaes_t *, double *rgx); 
const char *   cmaes_Test(cmaes_t *, const char *szIdentifier);
void           cmaes_UpdateEigensystem(cmaes_t *, int flgforce);
void           cmaes_WriteToFile(cmaes_t *, const char *szKeyWord,
                                 const char *szFileName); 
#endif
#if 0
/* --------------------------------------------------------- */
/* --------------- A Very Short Example -------------------- */
/* --------------------------------------------------------- */
#include "cmaes_interface.h"
double fitfun(double *x){ /* function "cigtab" to be minized */
  int i; 
  double sum = 1e4*x[0]*x[0] + 1e-4*x[1]*x[1];
  for(i = 2; i < x[-1]; ++i)  
    sum += x[i]*x[i]; 
  return sum;
}
int main() {
  double (*pFun)(double *) = &fitfun; /* any appropriate function pointer */
  cmaes_t evo;

  /* Initialize everything into the struct evo */
  cmaes_init(&evo, pFun, 0, NULL, NULL, 0, 0, "incmaes.par"); 

  /* Iterate until stop criterion holds */
  while(!cmaes_Test(&evo, "stop"))
    { 
      /* generate lambda new search points */
      cmaes_SampleDistribution(&evo, NULL);  
      /* evaluate the new search points using pfun */ 
      cmaes_EvaluateSample(&evo, NULL);      
      /* update the search distribution used for cmaes_SampleDistribution() */
      cmaes_ReestimateDistribution(&evo, NULL);  
      /* read from file and e.g. print output, or stop, etc */ 
      cmaes_ReadSignals(&evo, "signals.par");   
    }
  cmaes_WriteToFile(&evo, "all", "allcmaes.dat");  /* writes final results */
  cmaes_WriteToFile(&evo, "resume", "allcmaes.dat");   /* data for restart */
  cmaes_exit(&evo);
  return 0;
}
#endif
#if 0
/* --------------------------------------------------------- */
/* ----------- Rough, Incomplete Documentation ------------- */
/* --------------------------------------------------------- */
 
Remark: A number of input parameters have default values which
  can be invoked by the zero value (0 or NULL). 

cmaes_init(cma_struct_ptr, pFun, dimension, xstart, sigma, 
           seed, lambda, input_parameter_filename): 

    DEFAULTS of input parameters (invoked by 0):
        cma_struct_ptr           : Will be initialized here, not an input  
                                   value, no default available.  
        pFun                     : NULL
        dimension                : 0
        xstart                   : [0.5,...,0.5], N-dimensional vector. 
        sigma                    : [0.3,...,0.3], N-dimensional vector. 
        seed                     : random, see file actpar... 
        lambda                   : 4+(int)(3*log(N))
        input_parameter_filename : "incmaes.par"

    Input parameters: 
        cma_struct_ptr: Pointer to CMA-ES struct cmaes_t.

    Optional (default, induced by zero value): 
        pfun : Pointer to objective/fitness function to be 
            minimized. 

        dimension, int : Search space dimension N. Must be defined here
            or in the input parameter file. 

        xstart, double *: Initial point in search space. 

        sigma, double * : double array of size dimension
            N. Coordinatewise initial standard deviation of
            the sample distribution. The expected initial distance
            between xstart and the optimum per coordinate should not
            be much larger than sigma. 

        seed, int (randomly chosen by default): Random seed, written 
            to actparcmaes.par. 

        lambda, int : population size, number of sampled candidate 
            solutions per generation. 

        input_parameter_filename, char *: File which should be edited
            properly. Filename "non" omits reading and writing of any
            parameters in cmaes_init(), "writeonly" omits reading but
            still writes used parameters to file "actparcmaes.par".

    Details: Default values as shown above are invoked by zero values
        (0 or NULL). The dimension has to be defined >0 here or in the
        input parameter file ("incmaes.par"). pFun can be defined here
        or when calling cmaes_EvaluateSample(...,pFun).  pFun needs
        not to be defined if cmaes_EvaluateSample() is not used.

cmaes_resume_distribution(cmaes_t *evo_ptr, char *filename):
    Input parameters: 
        evo_ptr, cmaes_t *: Pointer to cma-es struct. 

        filename: A file, that was written presumably by
            cmaes_WriteToFile(evo_ptr, "resume", filename).

    Details: Allows to restart with saved distribution parameters (use
        cmaes_WriteToFile for saving). Keyword "resume" followed by a
        filename in incmaes.par invokes this function during
        initialization.  Searches in filename for the last occurence
        of "resume", followed by a dimension number, and reads the
        subsequent values for xmean, evolution paths ps and pc, sigma
        and covariance matrix.  Note that you have to call
        cmaes_init() before calling cmaes_resume_distribution()
        explicitely.  In the former all remaining
        (strategy-)parameters are set. It can be useful to edit the
        written parameters, in particular to increae sigma, before
        doing a restart.
 
cmaes_SampleDistribution(evo_ptr, xmean):
    Input parameters: 
      evo_ptr, cmaes_t *: Pointer to cma-es struct. 

    Optional: 
      xmean, const double *: Recent distribution mean. The default
          equals xstart in the first generation and the return value
          of cmaes_ReestimateDistribution() in the following
          generations.

    Return, double **: A pointer to a "population" of lambda
        N-dimensional samples.

cmaes_EvaluateSample(evo_ptr, pFun): 
    Input parameters: 
      evo_ptr, cmaes_t *: Pointer to cma-es struct. 

    Optional: 
        pFun, double(*)(double *)): Pointer to objective function. 

    Return, double *: Array of lambda function values. Evaluate lambda
        by calling cmaes_Get(evo_ptr, "lambda") or cmaes_Get(evo_ptr,
        "samplesize"). 
        
cmaes_ReestimateDistribution(evo_ptr, rgFuncValue):
    Input parameters: 
      evo_ptr, cmaes_t *: Pointer to cma-es struct. 

    Optional: 
        rgFuncValue, const double *: An array of lambda function
            values. By default the return value of
            cmaes_EvaluateSample().

    Return, double *:  Mean value of the new distribution. 

    Details: Core procedure of the CMA-ES algorithm. Sets a new mean
        value and estimates the new covariance matrix and a new step
        size for the normal search distribution. 

cmaes_Get(evo_ptr, szKeyWord): Returns the desired value. See 
    implementation in cmaes.c. 

cmaes_Getp(evo_ptr, szKeyWord): Returns a pointer to the desired vector 
    value (e.g. "xbestever"). See implementation in cmaes.c. 

cmaes_ReSampleSingle(evo_ptr, x): See example.c

cmaes_Test(keyword): 
    Input parameters: 
        "stop" is the only valid keyword at present. 

    Return value: NULL, if no stop criterion is fulfilled. Otherwise a
        string with the stop condition description. See example.c.

    Details: Some stopping criteria can be set in incmaes.par. Internal
        stopping criteria include a maximal condition number 10^14 for 
        the covariance matrix and situations where the numerical 
        discretisation error in x-space becomes noticable. 
                                      
cmaes_WriteToFile(evo_ptr, szKeyWord, szFileName):
    Input parameters: 
      evo_ptr, cmaes_t *: Pointer to cma-es struct. 

    Optional: 
        szKeyWord, const char *: There are a few keywords
            available. Most of them can be combined with a "+". In
            doubt confer to the implemtation in cmaes.c. Keywords: 

           "all": Writes a fairly complete status of the
               search. Missing is the actual random number generator
               status und the new coordinate system (matrix B).  
          
           "B": The complete basis B of normalized eigenvectors,
               sorted by the size of the corresponding eigenvalues.

           "eval": number of function evaluations so far. 

           "few": writes in one row: number of function evaluations;
               function value of best point in recent sample
               population; sigma; maximal standard deviation in
               coordinate axis; minimal standard deviation in
               coordinate axis; axis ratio of mutation ellipsoid;
               minimum of diagonal of D. 

           "few(diag(D))": 4 to 6 sorted eigen value roots, including the 
                 smallest and largest. 

           "resume": Writes dynamic parameters for reading with
               cmaes_resume_distribution. Alternatively (and more
               convenient) use keyword resume in incmaes.par for
               reading.

            further keywords: "dim", "funval", "N", "xbest",
                "xmean",...  See also implementation in cmaes.c.

         szFileName, const char *: File name, default is "tmpcmaes.dat".

    Usefull combined keywords can be "eval+funval", "eval+few(diag(D))"
    "few+few(diag(D))", "all+B". 

#endif


