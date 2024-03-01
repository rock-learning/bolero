/**
 * \file CMAESOptimizer.cpp
 * \author Malte Langosz
 * \brief An implementation of cmaes.
 *
 * Version 0.1
 */

#include <configmaps/ConfigData.h>

#include "CMAESOptimizer.h"
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>     // get time elapsed

using namespace configmaps;
//#define DEBUG_FOO

namespace bolero {
  namespace cmaes_optimizer {

    CMAESOptimizer::CMAESOptimizer(lib_manager::LibManager *theManager) :
      Optimizer(theManager, "cmaes_optimizer", 1) {
      rgFunVal = NULL;
      isInit = false;
      lambda = 0;
    }

    void CMAESOptimizer::init(int dimension, std::string config) {
      if(isInit){
        deinit();
      }

      assert(dimension > 0);
      seed = 0;
      char *seedChar = getenv("BL_SEED");
      if(seedChar) {
        sscanf(seedChar, "%ld", &seed);
      }
      if(seed == 0) {
        timeval t;
        gettimeofday(&t, NULL);
        unsigned long ms = t.tv_sec * 1000 + t.tv_usec / 1000;
        seed = ms+getpid()*1000;
      }
      std::string seedFilename;
      char *logPath = getenv("BL_LOG_PATH");
      if(logPath) {
         seedFilename = std::string(logPath) + "/seed.txt";
         logFileInd = std::string(logPath) + "/cmaes_fitness_ind.txt";
         logFileGen = std::string(logPath) + "/cmaes_fitness_gen.txt";
         logFileBest = std::string(logPath) + "/cmaes_fitness_best.txt";
      }
      else {
        logFileInd = "cmaes_fitness_ind.txt";
        logFileGen = "cmaes_fitness_gen.txt";
        logFileBest = "cmaes_fitness_best.txt";
        seedFilename = "seed.txt";
      }
      FILE *seedFile = fopen(seedFilename.c_str(), "w");
      if(seedFile) {
        fprintf(seedFile, "seed: %ld\n", seed);
        fclose(seedFile);
      }

      FILE *outFile = fopen(logFileInd.c_str(), "w");
      fclose(outFile);
      outFile = fopen(logFileGen.c_str(), "w");
      fclose(outFile);
      outFile = fopen(logFileBest.c_str(), "w");
      fclose(outFile);

      best = DBL_MAX;
      bestGen = DBL_MAX;
      bestGenIndex = 0;
      bestParams = NULL;

      this->dimension = dimension;

      double *xstart = new double[dimension];
      double *sigma = new double[dimension];

      logIndividual = logGeneration = logBest = false;
      reinitSigma = -1.;
      sigmaThreshold = -1.;

      double startSigma = 1.0;

      if(config != "")
      {
        ConfigMap map = ConfigMap::fromYamlString(config);

        if(map.hasKey("Optimizer")) {
            ConfigMap &m = map["Optimizer"];
            if(!lambda) {
            lambda = m.get("PopulationSize", lambda);
            }
            logIndividual = m.get("LogIndividual", false);
            logGeneration = m.get("LogGeneration", false);
            logBest = m.get("LogBest", false);
            reinitSigma = m.get("ReinitSigma", -1.);
            sigmaThreshold = m.get("SigmaThreshold", -1.);
            startSigma = m.get("StartSigma",1.0);

        }
      }

      for(int i=0; i<dimension; ++i) {
        xstart[i] = 0.5;
        sigma[i] = startSigma;
      }

      cmaes_init(&evo, NULL, dimension, xstart, sigma, seed, lambda, "non");
      lambda = evo.sp.lambda;
      rgFunVal = new double[lambda];

      // then generate the first generation
      rgx = cmaes_SampleDistribution(&evo, NULL);
      individual = 0;
      delete[] xstart;
      delete[] sigma;
      isInit = true;
    }

    void CMAESOptimizer::reinit(int dimension, int l, double *start) {
      if(l) lambda = l;
      if(!isInit) {
        init(dimension);
      }
      {
        deinit();
        best = DBL_MAX;
        bestGen = DBL_MAX;
        bestGenIndex = 0;
        bestParams = NULL;

        this->dimension = dimension;

        double *xstart = new double[dimension];
        double *sigma = new double[dimension];

        //#ifdef DEBUG_FOO
        //fprintf(stderr, "c%d ", dimension);
        //#endif
        for(int i=0; i<dimension; ++i) {
          if(start) {
            xstart[i] = start[i];
          }
          else {
            xstart[i] = 0.5;
          }
          if(reinitSigma > 0) sigma[i] = reinitSigma;
          else sigma[i] = 0.5;
        }

        cmaes_init(&evo, NULL, dimension, xstart, sigma, ++seed, lambda, "non");
        lambda = evo.sp.lambda;
        rgFunVal = new double[lambda];

        // then generate the first generation
        rgx = cmaes_SampleDistribution(&evo, NULL);
        individual = 0;
        delete[] xstart;
        delete[] sigma;
        isInit = true;
      }
    }

    void CMAESOptimizer::deinit() {
      if(isInit) {
        delete[] rgFunVal;
        rgFunVal = NULL;
        if(bestParams) delete[] bestParams;
        bestParams = NULL;
        isInit = false;
        cmaes_exit(&evo);
      }
    }

    CMAESOptimizer::~CMAESOptimizer() {
      if(isInit) {
        char *logDir = getenv("BL_LOG_PATH");
        if(logDir) {
          std::string file = logDir;
          file += "/allcmaes.dat";
          cmaes_WriteToFile(&evo, "all", file.c_str());
          file = logDir;
          file += "/resume.dat";
          cmaes_WriteToFile(&evo, "resume", file.c_str());
        }
        deinit();
      }

    }

    void CMAESOptimizer::getNextParameters(double *p, int numP) {
      assert(numP == dimension);

      for(int i=0; i<dimension; ++i) {
        p[i] = rgx[individual][i];
        saw(p+i, 0.0, 1.0);
      }
    }

    void CMAESOptimizer::getBestParameters(double *p, int numP) {
      assert(numP == dimension);

      for(int i=0; i<dimension; ++i) {
        if(bestParams) {
          p[i] = bestParams[i];
        }
        else {
          p[i] = 0.0;
        }
        saw(p+i, 0.0, 1.0);
      }
    }

    void CMAESOptimizer::setEvaluationFeedback(const double *feedbacks,
                                               int numFeedbacks) {
      double fitness = 0;

      if(numFeedbacks == 1) {
        fitness = feedbacks[0];
      }
      else {
        for(int i=0; i<numFeedbacks; ++i) {
          fitness += pow(feedbacks[i], 2);
        }
        fitness = sqrt(fitness);
      }

      if(fitness < bestGen) {
        bestGen = fitness;
        bestGenIndex = individual;
      }

      if(logIndividual) {
        FILE *outFile = fopen(logFileInd.c_str(), "a");
        fprintf(outFile, "%d %g", individual, fitness);
        for(int i=0; i<dimension; ++i) {
          fprintf(outFile, " %g", rgx[individual][i]);
        }
        fprintf(outFile, "\n");
        fclose(outFile);
      }
      if(fitness < best) {
        best = fitness;
        if(logBest) {
          FILE *outFile = fopen(logFileBest.c_str(), "a");
          fprintf(outFile, "%d %g", individual, fitness);
          for(int i=0; i<dimension; ++i) {
            fprintf(outFile, " %g", rgx[individual][i]);
          }
          fprintf(outFile, "\n");
          fclose(outFile);
        }
        if(!bestParams) bestParams = new double[dimension];
        memcpy(bestParams, rgx[individual], dimension*sizeof(double));
      }
      rgFunVal[individual] = fitness;
      if(++individual >= lambda) {
        if(logGeneration) {
          FILE *outFile = fopen(logFileGen.c_str(), "a");
          fprintf(outFile, "%d %g", bestGenIndex, bestGen);
          for(int i=0; i<dimension; ++i) {
            fprintf(outFile, " %g", rgx[bestGenIndex][i]);
          }
          fprintf(outFile, "\n");
          fclose(outFile);
        }
#ifdef DEBUG_FOO
        fprintf(stderr, "sig: %g ", evo.sigma);
#endif
        bestGen = DBL_MAX;
        cmaes_ReestimateDistribution(&evo, rgFunVal);
        cmaes_ReadSignals(&evo, (char*)"signals.par");
        rgx = cmaes_SampleDistribution(&evo, NULL);
        individual = 0;
        if(evo.sigma < 0.000000001 || evo.sigma > 100) {
          if(reinitSigma > 0) reinitSigma *= 2;
          double *start = new double[dimension];
          memcpy(start, bestParams, sizeof(double)*dimension);
          reinit(dimension, 0, start);
          delete[] start;
        }
      }
    }

    bool CMAESOptimizer::isBehaviorLearningDone() const {
        if(evo.sigma < sigmaThreshold) {
            return true;
        }
        return false;
    }

    void CMAESOptimizer::getNextParameterSet(double *p, int numP,
                                             int batchSize) const {
      assert(numP == dimension);
      assert(batchSize == lambda);


      for(int l=0; l<lambda; ++l) {
        if(l >= batchSize) break;
        for(int i=0; i<dimension; ++i) {
          if(i>=numP) break;
          p[l*dimension+i] = rgx[l][i];
          saw(p+l*dimension+i, 0.0, 1.0);
        }
      }
    }

    void CMAESOptimizer::setParameterSetFeedback(const double *feedback,
                                                 int numFeedbacksPerSet,
                                                 int batchSize) {
      assert(batchSize == lambda);

      for(size_t i=0; i<batchSize; ++i) {
        if(i>=lambda) break;
        setEvaluationFeedback(feedback+i*numFeedbacksPerSet,
                              numFeedbacksPerSet);
      }
    }

    int CMAESOptimizer::getBatchSize() const {
      return lambda;
    }

    void CMAESOptimizer::saw(double *val, double min, double max) const {
      double range = max - min;
      *val = fmod(fabs(*val), range*2);
      if((*val) > range) *val = range*2 - *val;
      *val += min;
    }

  } // end of namespace cmaes_optimizer
} // end of namespace bolero


DESTROY_LIB(bolero::cmaes_optimizer::CMAESOptimizer);
CREATE_LIB(bolero::cmaes_optimizer::CMAESOptimizer);
