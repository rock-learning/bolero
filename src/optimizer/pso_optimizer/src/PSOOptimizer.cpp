/**
 * \file PSOOptimizer.cpp
 * \author Lorenz Quack
 * \brief An implementation of Particle Swarm Optimization (PSO).
 *
 * Version 0.1
 */

#include "PSOOptimizer.h"

#include <configmaps/ConfigData.h>
#include <assert.h>
#include <limits>
#include <cstdlib> // for srand/rand
#include <ctime>   // for time(NULL) in srand
#include <cstring> // for memcpy
#include <cstdio>
#include <cmath>

using namespace configmaps;

namespace bolero {
  namespace pso_optimizer {
    inline void clamp(double *val, double minVal, double maxVal) {
      *val = (*val < minVal) ? minVal : ((*val > maxVal) ? maxVal : *val);
    }


    Particle::Particle(size_t dimension) {
      position = new double[dimension];
      velocity = new double[dimension];
      pMin = new double[dimension];
      pMinCost = std::numeric_limits<double>::max();

      for(size_t i = 0; i < dimension; ++i) {
        position[i] = double(rand()) / RAND_MAX;
        velocity[i] = 0;
        pMin[i] = position[i];
      }
    }

    Particle::~Particle() {
      delete[] position;
      delete[] velocity;
      delete[] pMin;
    }



    PSOOptimizer::PSOOptimizer(lib_manager::LibManager *theManager)
      : Optimizer(theManager, "pso_optimizer", 1)
      , wasInit(false) {
    }

    void PSOOptimizer::init(int dimension) {
      assert(dimension > 0);

      long seed = 0;
      char *seedChar = getenv("BL_SEED");
      if(seedChar) {
        sscanf(seedChar, "%ld", &seed);
        srand(seed);
      }
      if(seed == 0) {
        seed = time(NULL);
        srand(seed);
      }
      std::string seedFilename = ".";
      char *logPath = getenv("BL_LOG_PATH");
      if(logPath) {
         seedFilename = logPath;
      }
      seedFilename += "/seed.txt";
      FILE *seedFile = fopen(seedFilename.c_str(), "w");
      if(seedFile) {
        fprintf(seedFile, "seed: %ld\n", seed);
        fclose(seedFile);
      }

      this->dimension = dimension;
      particleCount = 4+(int)(3*log((double)dimension));

      ConfigMap map;
      ConfigMap *map2;
      map = ConfigMap::fromYamlFile("learning_config.yml");

      if(map.hasKey("BehaviorSearch Parameters")) {
        map2 = map["BehaviorSearch Parameters"];
        if(map2->find("PopulationSize") != map2->end()) {
          particleCount = (*map2)["PopulationSize"];
        }
      }

      particles = new Particle*[particleCount];
      for(int i = 0; i < particleCount; ++i) {
        particles[i] = new Particle(dimension);
      }
      gMin = new double[dimension];
      gMinCost = std::numeric_limits<double>::max();

      r = .5;
      wp = 1.5;
      wl = 1.5;
      wg = 1.5;

      generation = 0;
      individual = 0;
      wasInit = true;
    }

    PSOOptimizer::~PSOOptimizer() {
      if(wasInit) {
        for(int i = 0; i < particleCount; ++i) {
          delete particles[i];
        }
        delete[] particles;
        delete[] gMin;
      }
    }

    void PSOOptimizer::getNextParameters(double *p, int numP) {
      assert(numP == dimension);
      memcpy(p, particles[individual]->position, sizeof(double) * numP);
    }

    void PSOOptimizer::getBestParameters(double *p, int numP) {
      assert(numP == dimension);
      memcpy(p, gMin, sizeof(double) * numP);
    }

    void PSOOptimizer::setEvaluationFeedback(const double *rewards,
                                             int numRewards) {
      double fitness = 0;

      if(numRewards==1) {
        fitness = rewards[0];
      }
      else {
        for(int i=0; i<numRewards; ++i) {
          fitness += pow(rewards[i], 2);
        }
        fitness = sqrt(fitness);
      }

      if(particles[individual]->pMinCost > fitness) {
        particles[individual]->pMinCost = fitness;
        memcpy(particles[individual]->pMin, particles[individual]->position, sizeof(double) * dimension);
        if(gMinCost > fitness) {
          gMinCost = fitness;
          memcpy(gMin, particles[individual]->position, sizeof(double) * dimension);
        }
      }
      if(++individual >= particleCount) {
        updateParticles();
        individual = 0;
        generation++;
        fprintf(stdout, "Generation %3d's best fitness: %12.6f\n",
                generation, gMinCost);
        fprintf(stdout, "parameters: ");
        for(int i = 0; i < dimension; ++i) {
          fprintf(stdout, "%g, ", gMin[i]);
        }
        fprintf(stdout, "\n");
      }
    }

    std::vector<double*> PSOOptimizer::getNextParameterSet() const {
      std::vector<double*> parameterSet;
      double *p;

      for(int i=0; i<particleCount; ++i) {
        p = (double*)calloc(dimension, sizeof(double));
        memcpy(p, particles[i]->position, sizeof(double) * dimension);
        parameterSet.push_back(p);
      }
      return parameterSet;
    }

    void PSOOptimizer::setParameterSetFeedback(const std::vector<double> feedback) {
      std::vector<double>::const_iterator it;

      for(it=feedback.begin(); it!=feedback.end(); ++it) {
        setEvaluationFeedback(&(*it), 1);
      }
    }

    void PSOOptimizer::updateParticles() {
      for(int i = 0; i < particleCount; ++i) {
        Particle *p = particles[i];
        // update position
        for(int j = 0; j < dimension; ++j) {
          p->position[j] += p->velocity[j];
          clamp(&(p->position[j]), 0, 1);
        }
        // update velocity
        double sum = 0;
        for(int j = 0; j < dimension; ++j) {
          double rp = static_cast<double>(rand()) / RAND_MAX;
          double rg = static_cast<double>(rand()) / RAND_MAX;
          p->velocity[j] = (r * p->velocity[j] +
                            wp * rp * (p->pMin[j] - p->position[j]) +
                            wg * rg * (gMin[j] - p->position[j]));
          sum += fabs(p->velocity[j]);
        }
        if( sum / dimension < 0.0001) {
          for(int j = 0; j < dimension; ++j) {
            p->velocity[i] = double(rand()) / RAND_MAX;
          }
        }
      }
    }

  } // end of namespace pso_optimizer
} // end of namespace bolero


DESTROY_LIB(bolero::pso_optimizer::PSOOptimizer);
CREATE_LIB(bolero::pso_optimizer::PSOOptimizer);
