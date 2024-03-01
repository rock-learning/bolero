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
#include <sys/time.h>     // get time elapsed
#include <unistd.h>

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
      nr_reinits = 0;

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

    void PSOOptimizer::init(int dimension, std::string config) {
      assert(dimension > 0);

      if(wasInit) {
        deinit();
      }
      long seed = 0;
      char *seedChar = getenv("BL_SEED");
      if(seedChar) {
        sscanf(seedChar, "%ld", &seed);
      }
      if(seed == 0) {
        timeval t;
        gettimeofday(&t, NULL);
        long ms = t.tv_sec * 1000 + t.tv_usec / 1000;
        seed = ms+getpid()*1000;
      }
      std::string seedFilename = ".";
      char *logPath = getenv("BL_LOG_PATH");
      if(logPath) {
         seedFilename = logPath;
      }
      seedFilename += "/seed.txt";
      FILE *seedFile = fopen(seedFilename.c_str(), "w");
      if(seedFile) {
        fclose(seedFile);
      }
      srand(seed);
      this->dimension = dimension;
      particleCount = 4+(int)(3*log((double)dimension));

      maxReinits = -1;

      if(config != "")
      {
        ConfigMap map = ConfigMap::fromYamlString(config);
        ConfigMap *map2;

        if(map.hasKey("Optimizer")) {
            map2 = map["Optimizer"];
            if(map2->find("PopulationSize") != map2->end()) {
              particleCount = (*map2)["PopulationSize"];
            }
            map2 = map["Optimizer"];
            if(map2->find("MaxReinits") != map2->end()) {
              maxReinits = (*map2)["MaxReinits"];
            }
        }
      }

      particles = new Particle*[particleCount];
      for(int i = 0; i < particleCount; ++i) {
        particles[i] = new Particle(dimension);
        // set first particle to middle position
        if (i == 0) {
          for(size_t j = 0; j < dimension; ++j) {
            particles[0]->position[j] = 0.5;
            particles[0]->velocity[j] = 0;
            particles[0]->pMin[j] = particles[0]->position[j];
          }
        }
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
      minReinitsPerParticle = 0;
    }

    void PSOOptimizer::deinit() {
      if(wasInit) {
        for(int i = 0; i < particleCount; ++i) {
          delete particles[i];
        }
        delete[] particles;
        delete[] gMin;
        wasInit = false;
      }
    }

    PSOOptimizer::~PSOOptimizer() {
      if(wasInit) {
        char *logDir = getenv("BL_LOG_PATH");
        if(logDir) {
          std::string file = logDir;
          file += "/pso_best_params.dat";
          FILE *resFile = fopen(file.c_str(), "w");
          if(resFile) {
            for(int i=0;i<dimension;++i) {
              fprintf(resFile, "%g, ", gMin[i]);
            }
            fclose(resFile);
          }
        }
        deinit();
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
      }
    }

    bool PSOOptimizer::isBehaviorLearningDone() const {
      if (minReinitsPerParticle >= maxReinits && maxReinits >= 0) {
          return true;
      }
      return false;
    }


    void PSOOptimizer::getNextParameterSet(double *p, int numP, int batchSize) const {
      assert(numP == dimension);
      assert(batchSize == particleCount);

      for(int i=0; i<particleCount; ++i) {
        memcpy(p+i*numP, particles[i]->position, sizeof(double) * dimension);
      }
    }

    void PSOOptimizer::setParameterSetFeedback(const double *feedback,
                                               int numFeedbacksPerSet,
                                               int batchSize) {
      assert(batchSize == particleCount);

      for(size_t i=0; i<batchSize; ++i) {
        if(i>=particleCount) break;
        setEvaluationFeedback(feedback+i*numFeedbacksPerSet,
                              numFeedbacksPerSet);
      }
    }

    int PSOOptimizer::getBatchSize() const {
      return particleCount;
    }

    void PSOOptimizer::updateParticles() {
      minReinitsPerParticle = std::numeric_limits<int>::max();
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
            p->velocity[j] = double(rand()) / RAND_MAX;
          }
          ++p->nr_reinits;
        }
        minReinitsPerParticle = std::min(minReinitsPerParticle, p->nr_reinits);
      }
    }

  } // end of namespace pso_optimizer
} // end of namespace bolero


DESTROY_LIB(bolero::pso_optimizer::PSOOptimizer);
CREATE_LIB(bolero::pso_optimizer::PSOOptimizer);
