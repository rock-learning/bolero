/**
 * \file CMAESOptimizer.h
 * \author Malte Langosz
 * \brief An implementation of cmaes for the ml_interfaces::Optimizer.
 *
 * Version 0.1
 */

#ifndef __CMAES_OPTIMIZER_H__
#define __CMAES_OPTIMIZER_H__

#ifdef _PRINT_HEADER_
  #warning "CMAESOptimizer.h"
#endif

#include <Optimizer.h>

#include <string>

extern "C" {
#include "cmaes_interface.h"
}

namespace bolero {
  namespace cmaes_optimizer {

    class CMAESOptimizer : public Optimizer {

    public:
      CMAESOptimizer(lib_manager::LibManager *theManager);
      virtual ~CMAESOptimizer();

      CREATE_MODULE_INFO();

      virtual void init(int dimension, std::string config="");
      virtual void getNextParameters(double *p, int numP);
      virtual void getBestParameters(double *p, int numP);
      virtual void setEvaluationFeedback(const double *feedbacks,
                                         int numFeedbacks);
      virtual bool isBehaviorLearningDone() const;
      virtual std::vector<double*> getNextParameterSet() const;
      virtual void setParameterSetFeedback(const std::vector<double> feedback);
      void reinit(int dimension, int lambda=0, double *start=NULL);
      int getDimension() {return dimension;}

    private:
      cmaes_t evo;
      double *rgFunVal;
      double *const*rgx;
      int lambda;
      int individual;
      double best, bestGen;
      int bestGenIndex;
      double *bestParams;
      bool isInit, logIndividual, logGeneration, logBest;
      double reinitSigma;
      double sigmaThreshold;
      unsigned long seed;
      std::string logFileInd, logFileGen, logFileBest;

      void saw(double *val, double min, double max) const;
      void deinit();
    }; // end of class definition CMAESOptimizer

  } // end of namespace cmaes_optimizer
} // end of namespace bolero

#endif // __CMAES_OPTIMIZER_H__
