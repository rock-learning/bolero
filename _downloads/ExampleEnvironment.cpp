/**
 * \file ExampleEnvironment.h
 * \author Sebastian Klemp
 * \brief see header
 *
 * Version 0.1
 */


#include "ExampleEnvironment.h"

#include <cfloat>

#include <mars/interfaces/sim/MarsPluginTemplate.h>
#include <mars/interfaces/sim/MotorManagerInterface.h>
#include <mars/interfaces/sim/SensorManagerInterface.h>

using namespace configmaps;

namespace bolero {
  namespace Example_environment {

    ExampleEnvironment::ExampleEnvironment(lib_manager::LibManager *theManager)
    : Environment(theManager, "Example_environment", 1.0),
      MARSEnvironment(theManager, "Example_environment", 1.0),
      MAX_TIME(20000),
      numJoints(7),
      numAllJoints(7){
    }
  
    void ExampleEnvironment::initMARSEnvironment() {
      // Here we use the environment variable "ROCK_CONFIGURATION_PATH" in
      // order to "find" the scene file to be loaded. During installation it
      // should be put in "$ROCK_CONFIGURATION_PATH/aila_environment".
      const char *configPath = getenv("ROCK_CONFIGURATION_PATH");
      if (strcmp(configPath, "") == 0) {
        fprintf(stderr, "WARNING: The ROCK_CONFIGURATION_PATH is not set! Did you \"source env.sh\"?\n");
      }
      std::string sceneFile = std::string(configPath) + \
                              "/example_environment/example.scn";
      bool velocityControl = false;

      // check for environment parameters
      configmaps::ConfigMap map = ConfigMap::fromYamlFile("learning_config.yml");
      if(map.find("Environment Parameters") != map.end()) {
        ConfigMap *map2 = &(map["Environment Parameters"][0].children);
        // check if the environment should use velocity or position controlled
        // motors
        if(map2->find("velocityControl") != map2->end()) {
          velocityControl = (*map2)["velocityControl"][0].getBool();
        }
      }

      if(velocityControl) {
        sceneFile = std::string(configPath) + "/example_environment/example_vc.scn";
      }

      control->sim->loadScene(sceneFile.c_str());

      resetMARSEnvironment();
    }

    void ExampleEnvironment::resetMARSEnvironment() {
      // get the IDs from the sensors for the outputsvalues
      getSensorIDs();
      // get the IDs from the motors for the inputs
      getMotorIDs();
      fitness = 0.0;
      evaluation_done = false;
    }

    void ExampleEnvironment::handleMARSError() {
      fitness = DBL_MAX;
      evaluation_done = true;
    }

    void ExampleEnvironment::getSensorIDs() {
      sensorIDs.clear();
      std::vector<mars::interfaces::core_objects_exchange>::iterator jIter;
      std::vector<mars::interfaces::core_objects_exchange> sensorList;
      // get a list of all sensors to search for the right ones
      control->sensors->getListSensors(&sensorList);
      for(jIter = sensorList.begin(); jIter != sensorList.end(); ++jIter) {
        if(jIter->name  == "Motor_Angles"){
          // add the name and index of the sensor to the map
          sensorIDs.push_back(jIter->index);
          break;
        }
      }
      for(jIter = sensorList.begin(); jIter != sensorList.end(); ++jIter) {
        if(jIter->name  == "Endeffector_position"){
          // add the name and index of the sensor to the map
          sensorIDs.push_back(jIter->index);
          break;
        }
      }
      for(jIter = sensorList.begin(); jIter != sensorList.end(); ++jIter) {
        if(jIter->name  == "Endeffector_rotation"){
          // add the name and index of the sensor to the map
          sensorIDs.push_back(jIter->index);
          break;
        }
      }
      for(jIter = sensorList.begin(); jIter != sensorList.end(); ++jIter) {
        if(jIter->name  == "Endeffector_velocity"){
          // add the name and index of the sensor to the map
          sensorIDs.push_back(jIter->index);
          break;
        }
      }
    }

    void ExampleEnvironment::getMotorIDs() {
      motorIDs.clear();
      std::vector<mars::interfaces::core_objects_exchange>::iterator jIter;
      std::vector<mars::interfaces::core_objects_exchange> motorList;
      // get a list of all motors to search for the right ones
      control->motors->getListMotors(&motorList);
      for(jIter = motorList.begin(); jIter != motorList.end(); ++jIter) {
        motorIDs.push_back(jIter->index);
      }
    }

    int ExampleEnvironment::getNumInputs() const {
      return numJoints;
    }

    int ExampleEnvironment::getNumOutputs() const {
      // 7 joint angles + 3 for endeffector_position, endeffector_rotation, endeffector_velocity each
      return numJoints + 9;
    }

    void ExampleEnvironment::createOutputValues(void) {

      mars::interfaces::sReal *data = NULL;
      unsigned int sensorIndex = 0;

      if(sensorIDs.size() > 0) {

        // get the data from the Motor_Angles sensor
        unsigned data_size = control->sensors->getSensorData(sensorIDs[sensorIndex],&data);
        if(data_size == numAllJoints){
          for(unsigned int j = 0;j<data_size;j++){
            outputs[j] = data[j];
          }
        }
        free(data);
        data = NULL;
      }
      if(sensorIDs.size() > 1) {
        unsigned int startIndex = numJoints+sensorIndex*3;
        sensorIndex++;
        // get the data from the endeffector_position sensor
        unsigned data_size = control->sensors->getSensorData(sensorIDs[sensorIndex],&data);
        if(data_size == 3){
          for(unsigned int j = 0;j<data_size;j++){
            outputs[startIndex+j] = data[j];
          }
        }
        free(data);
        data = NULL;
      }
      if(sensorIDs.size() > 2) {
        unsigned int startIndex = numJoints+sensorIndex*3;
        sensorIndex++;
        // get the data from the endeffector_rotation sensor
        unsigned data_size = control->sensors->getSensorData(sensorIDs[sensorIndex],&data);
        if(data_size == 3){
          for(unsigned int j = 0;j<data_size;j++){
            outputs[startIndex+j] = data[j];
          }
        }
        free(data);
        data = NULL;
      }
      if(sensorIDs.size() > 3) {
        unsigned int startIndex = numJoints+sensorIndex*3;
        sensorIndex++;
        // get the data from the endeffector_velocity sensor
        unsigned data_size = control->sensors->getSensorData(sensorIDs[sensorIndex],&data);
        if(data_size == 3){
          for(unsigned int j = 0;j<data_size;j++){
            outputs[startIndex+j] = data[j];
          }
        }
        free(data);
        data = NULL;
      }
      
      // calculate the fitness (does the motors have reached the desired values)
      for(unsigned int i=0;i<numJoints;i++){
        fitness += inputs[i]-outputs[i];
      }

      // test if the vehicle is out of time
      if (leftTime > MAX_TIME) {
        fitness = DBL_MAX;
        evaluation_done = true;
        //fprintf(stderr, "fitness = %f\n", fitness);
      }
    }

    void ExampleEnvironment::handleInputValues() {
      bool set_values = true;
      unsigned int i;
      int tmp;

      for(i = 0; i < motorIDs.size(); i++) {
        if((tmp = std::fpclassify(inputs[i]))) {
          switch(tmp) {
          case FP_INFINITE:
            //LOG_ERROR("vals[%d] is infinite", i);
            set_values = false;
            break;
          case FP_NAN:
            //LOG_ERROR("vals[%d] is not a number", i);
            set_values = false;
            break;
          case FP_NORMAL:
            //LOG_ERROR("vals[%d] is normalized", i);
            break;
          case FP_SUBNORMAL:
            //LOG_ERROR("vals[%d] is denormalized %g", i, inputs[i]);
            break;
          case FP_ZERO:
            inputs[i] = 0.0;
            //LOG_ERROR("vals[%d] is zero", i);
            break;
          }
        }
      }
      if(set_values) {
        // set the motors to the given input values
        control->motors->setMotorValue(motorIDs[0], inputs[0]);
        control->motors->setMotorValue(motorIDs[1], inputs[1]);
        control->motors->setMotorValue(motorIDs[2], inputs[2]);
        control->motors->setMotorValue(motorIDs[3], inputs[3]);
        control->motors->setMotorValue(motorIDs[4], inputs[4]);
        control->motors->setMotorValue(motorIDs[5], inputs[5]);
        control->motors->setMotorValue(motorIDs[6], inputs[6]);
      }
    }

    bool ExampleEnvironment::isEvaluationDone() const {
      return evaluation_done;
    }

    int ExampleEnvironment::getFeedback(double *feedback) const {
      feedback[0] = fitness;
      return 1;
    }

  } // end of namespace Example_environment
} // end of namespace bolero

DESTROY_LIB(bolero::Example_environment::ExampleEnvironment);
CREATE_LIB(bolero::Example_environment::ExampleEnvironment);
