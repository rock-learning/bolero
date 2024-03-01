/**
 * \file PSOOptimizer.h
 * \author Lorenz Quack
 * \brief An implementation of Particle Swarm Optimization (PSO).
 *
 * Version 0.1
 */

#ifndef __PSO_OPTIMIZER_H__
#define __PSO_OPTIMIZER_H__

#ifdef _PRINT_HEADER_
  #warning "PSOOptimizer.h"
#endif

#include <Optimizer.h>

namespace bolero {
  namespace pso_optimizer {

    class Particle {
    public:
      Particle(size_t dimensions);
      ~Particle();
      double *position;
      double *velocity;
      double *pMin;
      double pMinCost;
      int nr_reinits;
    };

    class PSOOptimizer : public Optimizer {

    public:
      PSOOptimizer(lib_manager::LibManager *theManager);
      virtual ~PSOOptimizer();

      CREATE_MODULE_INFO();

      virtual void init(int dimension, std::string config="");
      virtual void getNextParameters(double *p, int numP);
      virtual void getBestParameters(double *p, int numP);
      virtual void setEvaluationFeedback(const double *feedbacks,
                                         int numFeedbacks);

      bool isBehaviorLearningDone() const;

      virtual void getNextParameterSet(double *p, int numP,
                                       int batchSize) const;
      virtual void setParameterSetFeedback(const double *feedback,
                                           int numFeedbacksPerSet,
                                           int batchSize);
      virtual int getBatchSize() const;


    private:
      void updateParticles();
      void deinit();

      int individual;
      int generation;
      int particleCount;
      Particle **particles;
      double *gMin;
      double gMinCost;
      double r, wp, wl, wg;
      bool wasInit;
      int minReinitsPerParticle;
      int maxReinits;

    }; // end of class definition PSOOptimizer

  } // end of namespace pso_optimizer
} // end of namespace bolero

#endif // __PSO_OPTIMIZER_H__
