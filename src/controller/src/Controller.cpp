#include "Controller.h"


#include <configmaps/ConfigData.h>
#include <BehaviorSearch.h>
#include <Environment.h>
#include <BLLoader.h>

#include <cassert>
#include <float.h>
#include <limits>
#include <signal.h>

using namespace lib_manager;
using namespace bolero;
using configmaps::ConfigMap;
using std::string;
using std::vector;

namespace bolero {

  bool Controller::exitController = false;

  bool checkFile(std::string file) {
    FILE *f = fopen(file.c_str(), "r");
    if(f) {
      fclose(f);
      return true;
    }
    return false;
  }

  void exitHandler(int sig) {
    fprintf(stderr, "want to exit controller\n");
    Controller::exitController = true;
  }

  int Controller::run() {
    BehaviorSearch *behaviorSearch;
    Environment *environment;
    Behavior *behavior;
    const char *blLogPath;
    const char *blConfPath;
    std::string confFile;
    double* feedbacks = new double[100];
    unsigned int num_feedbacks = 0; //, size_feedbacks = 0;
    double feedback;
    double minFeedback = DBL_MAX;
    double minTestFeedback = DBL_MAX;
    int minRun;
    int minTestRun = 0;
    FILE *fitnessLog = NULL;
    FILE *testFitnessLog = NULL;
    bool testMode = false;
    int testEveryXRun = 0;
    double epsilon = 0.000000001; // for testing new fitness values

#ifndef WIN32
    signal(SIGQUIT, exitHandler);
#endif
    signal(SIGABRT, exitHandler);
    signal(SIGTERM, exitHandler);
    signal(SIGINT, exitHandler);

    if(!(blLogPath = getenv("BL_LOG_PATH"))) {
      fprintf(stdout, "WARNING: No log path for results given! Using \".\" instead.\n");
      blLogPath = ".";
    }

    if(!(blConfPath = getenv("BL_CONF_PATH"))) {
      fprintf(stdout, "WARNING: No config path given! Using \".\" instead.\n");
      blConfPath = ".";
    }

    std::string libFile = string(blConfPath) + "/learning_libraries.txt";
    bool haveLibFile = checkFile(libFile);
    bl_loader::BLLoader *blLoader = new bl_loader::BLLoader();
    if(haveLibFile) {
      blLoader->loadConfigFile(libFile);
    }

    ConfigMap map = ConfigMap::fromYamlFile("learning_config.yml");
    string strEnvironment = map["Environment"]["type"];
    string strBehaviorSearch = map["BehaviorSearch"]["type"];
    int maxEvaluations = map["Controller"]["MaxEvaluations"];
    bool logAllBehaviors = false;
    bool evaluateExperiment = false;
    bool logResults = false;
    string experimentDir;

    if(map["Controller"].hasKey("LogAllBehaviors")) {
      logAllBehaviors = map["Controller"]["LogAllBehaviors"];
    }

    blLoader->loadLibrary(strEnvironment);
    blLoader->loadLibrary(strBehaviorSearch);

    if(map["Controller"].hasKey("GenerateFitnessLog")) {
      if(map["Controller"]["GenerateFitnessLog"]) {
        string fLogFilename = string(blLogPath) + "/fitness.txt";
        fitnessLog = fopen(fLogFilename.c_str(), "w");
        fLogFilename = string(blLogPath) + "/test_fitness.txt";
        testFitnessLog = fopen(fLogFilename.c_str(), "w");
      }
    }
    if(map["Controller"].hasKey("LogResults")) {
      logResults = map["Controller"]["LogResults"];
    }
    if(map["Controller"].hasKey("EvaluateExperiment")) {
      evaluateExperiment = map["Controller"]["EvaluateExperiment"];
    }
    if(map["Controller"].hasKey("EvaluatePathToExperiment")) {
      experimentDir << map["Controller"]["EvaluatePathToExperiment"];
    }

    if(map["Controller"].hasKey("TestEveryXRun")) {
      testEveryXRun = map["Controller"]["TestEveryXRun"];
      fprintf(stderr, "testevery: %d\n", testEveryXRun);
    }

    environment = blLoader->acquireEnvironment(strEnvironment);
    behaviorSearch = blLoader->acquireBehaviorSearch(strBehaviorSearch);

    assert(environment);
    assert(behaviorSearch);

    environment->init();
    //environment->init(map["Environment"][0].children.toYamlString());
    int numInputs = environment->getNumOutputs();
    int numOutputs = environment->getNumInputs();

    double *inputs = new double[numInputs];
    double *outputs = new double[numOutputs];
    fprintf(stderr, "num in- and outputs: %d %d\n", numInputs, numOutputs);
    behaviorSearch->init(numInputs, numOutputs);
    // behaviorSearch->init(numInputs, numOutputs, map["BehaviorSearch"][0].children.toYamlString());

    blLoader->dumpTo(string(blLogPath) + "/libs_info.xml");
    int evaluationCount = 0;
    do {
      if(evaluateExperiment) {
        behavior = behaviorSearch->getBehaviorFromResults(experimentDir);
      }
      else {
        if(testMode) {
          behavior = behaviorSearch->getBestBehavior();
        }
        else {
          behavior = behaviorSearch->getNextBehavior();
        }
      }

      do {
        environment->getOutputs(inputs, numInputs);
        behavior->setInputs(inputs, numInputs);
        behavior->step();
        behavior->getOutputs(outputs, numOutputs);
        environment->setInputs(outputs, numOutputs);
        environment->stepAction();
      } while(!environment->isEvaluationDone() && !exitController);
      if(exitController) break;
      /* Feedback interface need to be changed for better controller
         implementation.
      */
      /*
      num_feedbacks = environment->getNumFeedbacks();
      if(size_feedbacks != num_feedbacks) {
        if(size_feedbacks > 0) {
          delete[] feedbacks;
        }
        feedbacks = new double[num_feedbacks];
        size_feedbacks = num_feedbacks;
      }
      */
      num_feedbacks = environment->getFeedback(feedbacks);
      //assert(num_feedbacks == size_feedbacks);
      feedback = 0.0;
      for(int i = 0; i < num_feedbacks; i++)
        feedback += feedbacks[i];
      if(!testMode) {
        if(fitnessLog || logResults) {
          if(feedback < minFeedback-epsilon) {
            minFeedback = feedback;
            minRun = evaluationCount+1;
            if(fitnessLog) {
              fprintf(fitnessLog, "%d %g\n", evaluationCount, feedback);
            }
            if(logResults) {
              behaviorSearch->writeResults(blLogPath);
            }
          }
          else if(logAllBehaviors) {
            behaviorSearch->writeResults(blLogPath);
          }
        }
        behaviorSearch->setEvaluationFeedback(feedbacks, num_feedbacks);
        if(testEveryXRun > 0) {
          if(evaluationCount%testEveryXRun == 0) {
            testMode = true;
            environment->setTestMode(true);
          }
        }
      }
      else {
        if(testFitnessLog) {
          if(feedback < minTestFeedback-epsilon) {
            minTestFeedback = feedback;
            minTestRun = evaluationCount+1;
            fprintf(testFitnessLog, "%d %g\n", evaluationCount, feedback);
          }
        }
        testMode = false;
        environment->setTestMode(false);
      }

      if(evaluationCount % 100 == 0) {
        string filename = string(blLogPath) + "/learning_progress.txt";
        FILE *pFile = fopen(filename.c_str(), "w");
        fprintf(pFile, "number evaluations: %d\n", evaluationCount);
        fclose(pFile);
      }

      environment->reset();

      if(!testMode) {
        ++evaluationCount;
      }

    } while(evaluationCount < maxEvaluations &&
            !behaviorSearch->isBehaviorLearningDone() &&
            !environment->isBehaviorLearningDone() &&
            !exitController);

    behaviorSearch->writeResults(blLogPath);
    if(fitnessLog) {
      fclose(fitnessLog);
      string fLogFilename = string(blLogPath) + "/bestFitness_Controller.txt";
      fitnessLog = fopen(fLogFilename.c_str(), "w");
      fprintf(fitnessLog, "%d %g", minRun, minFeedback);
      fclose(fitnessLog);
    }

    if(testFitnessLog) {
      fclose(testFitnessLog);
      string fLogFilename = string(blLogPath) + "/bestTestFitness_Controller.txt";
      fitnessLog = fopen(fLogFilename.c_str(), "w");
      fprintf(fitnessLog, "%d %g", minTestRun, minTestFeedback);
      fclose(fitnessLog);
    }

    delete[] inputs;
    delete[] outputs;

    try {
      blLoader->releaseLibrary(strEnvironment);
    } catch(std::runtime_error e) {
      std::cout << e.what() << std::endl;
    }
    try {
      blLoader->releaseLibrary(strBehaviorSearch);
    } catch(std::runtime_error e) {
      std::cout << e.what() << std::endl;
    }

    delete blLoader;

    return 0;
  }

} /* end of namespace bolero */
