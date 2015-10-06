------------------------------------------------------------------
For the general purpose of this software see end of file. 

Delivered (ten) Files:
  README.txt : This file
  cmaes_interface.h : User interface header. Probably the first
      place to go after this file. It includes a very short 
      example and a rough DOCUMENTATION.  
  example.c : Example source code, its purpose is to be edited/extended. 
  example1.c : Example source code, only slight changes compared to
      example.c. example1.c does not read input parameters from
      an parameter file and output is reduced. 
  cmaes.h : Header, e.g. declaration of struct cmaes_t.  
  cmaes.c : Source code. 
  incmaes.par : Parameters to be read by the cmaes, e.g. problem
      dimension, initial search point and initial standard deviation. 
      This file should be edited.
  signals.par : File for controlling the running program. Printing 
      or writing to a file/console can be set on/off while the program 
      is running. Regular termination can be forced.
  plotcmaesdat.m : Plots files rescmaes.dat and xcmaes.dat in MATLAB. 
  plotcmaesdat.R : Plots files rescmaes.dat and xcmaes.dat in R. 


Files you may need to edit:
  example.c:  Plug in the objective function (pointer) that
    should be minimized.
  incmaes.par: Parameter file for changing e.g. initial values and
    stopping criteria without recompiling example.c. 
  signals.par: File to control the output *during runtime*. 


Output files written by cmaes_t: 
  actparcmaes.par : Parameters as actually used by the program. The
                    actual parameter setting is appended to the file  
                    after each start of the cmaes. 
  errcmaes.err  : Error messages. 

Output files written in example.c: 
  rescmaes.dat  : Output of function values etc. For plotting, see
                  plotcmaesdat.* and argument "few" and "few(diag(D))"
                  for function cmaes_WriteToFile in file
                  cmaes_interface.h. Note that all data are
                  appended. To see a single plot you have to move or
                  delete rescmaes.dat before starting the experiment.
  xcmaes.dat    : Output of x-variable vectors for plotting. See also 
                  rescmaes.dat.
  allcmaes.dat  : More output information.

------------------------------------------------------------------
How to start:

  0) Set up a working directory and (under Linux) do 
     "tar -xf cmaes_c.tar" within this directory or copy the unpacked
     files (at least the *.c, *.h, and *.par) into that directory.

  1) Compile and run the example program. Compilation e.g. with 
     the GNU c-compiler:
	"gcc -Wall -lm -o evo.exe cmaes.c example.c "
     and run with "evo.exe" or "./evo.exe". Take a look at the output. 

  2a) (optional) Invoke matlab (or R) and type plotcmaesdat (or
     plotcmaesdat()).  (You need to have the file plotcmaesdat.m (or
     plotcmaesdat.R) and the output data files in the working
     directory). You get a nice plot of the executed run. Remove files
     rescmaes.dat and xcmaes.dat to get a clean plot for the next
     execution (output is appended).

  2b) (optional) Change (increase) problem dimension and/or problem
     number in file incmaes.par and re-run.

  2c) (optional) Change problem dimension in incmaes.par to 300 and
     change output verbosity via file signals.par while the program
     is running: change e.g. "print fewinfo 200" into "print fewinfo
     -200" *and back*. Read comments. 

  2d) Change back problem dimension.  

  3) Take five minutes to look at "A Very Short Example" in file
     cmaes_interface.h. You might also refer to the documentation
     provided in the same file. 

  4) Now you are ready to inspect and edit example.c to plug in the
     function you want to optimize. Recompile afterwards. 

  5) Check "obligatory settings" part in incmaes.par regarding your
     function. Make sure that the scale of all objective parameter
     components of the function is somewhat similar and sigma
     corresponds to about half of the respective search intervals.

  6) If you want to use the output files, it is convenient to have a
     batch file that (re-)moves output files before next execution,
     like 
       mv rescmaes.dat xcmaes.dat OldFilesDirectory 
       ./evo.exe


------------------------------------------------------------------
TUTORIAL:
	http://www.tu-berlin.de/user/niko/cmatutorial.pdf

REFERENCES:
	http://www.tu-berlin.de/user/niko/publications.html

Hansen, N, and S. Kern (2004).  Evaluating the CMA Evolution Strategy
on Multimodal Test Functions. In: Eighth International Conference on
Parallel Problem Solving from Nature PPSN VIII, Proceedings,
pp. 282-291, Berlin: Springer

Hansen, N., S.D. Müller and P. Koumoutsakos (2003): Reducing the Time
Complexity of the Derandomized Evolution Strategy with Covariance
Matrix Adaptation (CMA-ES). Evolutionary Computation, 11(1).

Hansen, N. and A. Ostermeier (2001). Completely Derandomized
Self-Adaptation in Evolution Strategies. Evolutionary Computation,
9(2), pp. 159-195.

Hansen, N. and A. Ostermeier (1996). Adapting arbitrary normal
mutation distributions in evolution strategies: The covariance matrix
adaptation. In Proceedings of the 1996 IEEE International Conference
on Evolutionary Computation, pp. 312-317. 

------------------------------------------------------------------

GENERAL PURPOSE: 
The CMA-ES (Evolution Strategy with Covariance Matrix Adaptation) is a
robust search/optimization method. The goal is to minimize a given
objective function, f: R^n -> R.  The CMA-ES should be applied, if
e.g. BFGS and/or conjucate gradient methods fail due to a rugged
search landscape (e.g. discontinuities, outliers, noise, local optima,
etc.). Learning the covariance matrix in the CMA-ES is somewhat
equivalent to learning the inverse Hessian matrix in a quasi-Newton
method. On smooth landscapes the CMA-ES is roughly ten times slower
than BFGS, assuming derivatives are not directly available. For up to
N=10 parameters the simplex direct search method (Nelder & Mead) is
often faster, but less robust than CMA-ES.  On considerably hard
problems the search (a single run) is expected to take at least 30*N^2
to 300*N^2 function evaluations.

SOME MORE COMMENTS: 
The adaptation of the covariance matrix (e.g. by the CMA) is
equivalent to a general linear transformation of the problem
coding. Nevertheless every problem specific knowlegde about the best
linear transformation should be exploited before starting the search
procedure and an appropriate a priori transformation should be applied
to the problem. This also makes starting with the identity matrix as
covariance matrix the best choice.

