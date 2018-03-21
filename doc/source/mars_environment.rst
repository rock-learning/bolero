.. _mars_environment:

================
MARS Environment
================

This guide will explain how to set up a MARS-based environment for BOLeRo.
We published an example for a MARS environment in a
`separate project <https://github.com/rock-learning/throwing_environment>`_.
We will go through this example step by step here. In the end you will see
something like this:

.. raw:: html

    <center><img src="_static/throwing_environment.png" width=500px /></center>


Folder and File Structure of an Environment
===========================================

The structure of a project that provides a MARS-based environment always looks
very similar to this one.

**throwing_environment/**

* README.md - provides information for the user (installation, license, ...)
* CMakeLists.txt - defines how we can build this project with
  `CMake <http://www.cmake.org/>`_
* manifest.xml - defines dependencies for `Rock <http://rock-robotics.org>`_ or
  `ROS <http://www.ros.org/>`_
* throwing_environment.pc.in - defines package information like the location of
  the library, compiler flags etc.
* configuration/ - contains configuration files for the environmet

  * throwing.smurfs - description of the scene, tells MARS which objects it
    should load and how they are connected
  * robot - contains configuration files for the object with the name 'robot'

    * urdf - contains the URDF file of the object
    * smurf - contains specific configuration files for the simulation, for
      example, motor configuration and collision configuration

* src/ - contains source files for the environment implementation

The hardest part of creating a new simulation environment is usually to define
the scene (everything that is located in configuration/). We cannot solve this
problem here. Instead, we can refer to
`Phobos <https://github.com/rock-simulation/phobos>`_. Phobos is a plugin for
`Blender <https://www.blender.org/>`_. It enables the creation and modification
of WYSIWYG robot models and those can be exported to MARS scenes (and other
formats). We will take a closer look at the implementation of the scene with
existing configuration files (everything that is located in src/).


Building an Environment
=======================

Building a MARS-based environment is straightforward and not much different
to the standard CMake-based build process:


.. code-block:: bash

    mkdir build
    cd build
    cmake_debug ..  # this is different, 'cmake_debug' knows where bolero-dev is
    make install

We do not use `cmake ..` because we want to install the environment to our
BOLeRo installation. `cmake_debug` and `cmake_release` do automatically
set the correct `CMAKE_INSTALL_PREFIX`.


Creating Your Own Environment
=============================

We will show and explain important parts of
`this environment <https://github.com/rock-learning/throwing_environment>`_.
For further details, please take a look at the full source code.

CMakeLists.txt
--------------

A MARS environment depends on BOLeRo and the base class for MARS environments.

.. code-block:: cmake

    pkg_check_modules(BOLERO "bolero")
    include_directories(${BOLERO_INCLUDE_DIRS})

    pkg_check_modules(MARS_ENV "mars_environment")
    include_directories(${MARS_ENV_INCLUDE_DIRS})
    link_directories(${MARS_ENV_LIBRARY_DIRS})

The environment will be compiled to a library.

.. code-block:: cmake

    set(SOURCES src/ThrowingEnvironment.cpp)
    set(HEADERS src/ThrowingEnvironment.h)

    add_library(${PROJECT_NAME} SHARED ${SOURCES})

    target_link_libraries(${PROJECT_NAME}
        ${MARS_ENV_LIBRARIES}
        ${MARS_UTILS_LIBRARIES}
        ${CONFIGMAPS_LIBRARIES})

The following files have to be installed: the library, headers, pkg-config
information, and scene configuration files.

.. code-block:: cmake

    # Install the library into the lib folder
    install(TARGETS ${PROJECT_NAME} ${_INSTALL_DESTINATIONS})

    # Install headers into mars include directory
    install(FILES ${HEADERS} DESTINATION include/bolero/${PROJECT_NAME})

    # Prepare and install necessary files to support finding of the library 
    # using pkg-config
    configure_file(${PROJECT_NAME}.pc.in ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc @ONLY)
    install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc DESTINATION lib/pkgconfig)

    install(FILES configuration/throwing.smurfs
            DESTINATION configuration/${PROJECT_NAME}/)
    install(DIRECTORY configuration/robot
            DESTINATION configuration/${PROJECT_NAME})
    install(DIRECTORY configuration/target
            DESTINATION configuration/${PROJECT_NAME})

ThrowingEnvironment.h
---------------------

Our `ThrowingEnvironment` is a subclass of `mars_environment::MARSEnvironment`
and, in this case, BOLeRo's `ContextualEnvironment`. A contextual environment
defines a problem that can be parameterized by a context vector.

.. code-block:: c++

    namespace bolero {
      namespace throwing_environment {

        class ThrowingEnvironment : public mars_environment::MARSEnvironment,
                                    public ContextualEnvironment {

The following functions have to be implemented.

.. code-block:: c++

          virtual void initMARSEnvironment();
          virtual void resetMARSEnvironment();
          virtual void handleMARSError();

          virtual int getNumInputs() const;
          virtual int getNumOutputs() const;

          virtual void createOutputValues();
          virtual void handleInputValues();

          virtual int getFeedback(double *feedback) const;

          bool isEvaluationDone() const;
          bool isBehaviorLearningDone() const { return false; }

          virtual double* request_context(double *context,int size);
          virtual int get_num_context_dims() const;

ThrowingEnvironment.cpp
-----------------------

In `ThrowingEnvironment::initMARSEnvironment()` we load the configuration of
the environment.

.. code-block:: c++

        void ThrowingEnvironment::initMARSEnvironment()
        {
          readConfig();
          request_context(targetPos.data(), 2);

          if(!isSceneLoaded)
          {
            std::string sceneFile = getConfigPath() +
                "/throwing_environment/throwing.smurfs";
            control->sim->loadScene(sceneFile.c_str());
            isSceneLoaded = true;
          }

          getMotorIDs();
        }

        void ThrowingEnvironment::readConfig()
        {
          // Parameters of the environment are in the file "learning_config.yml".
          // It should be located in the current working directory. This
          // environment accepts the additional parameters
          // - ballThrowTime: after this has been reached (number of time steps),
          //   the ball will be detached from the robot
          // - armHeight: simulates that the arm is mounted on a table, this is the
          //   height of the table, the simulation stops when the ball hits the
          //   virtual ground
          // - verbose - verbosity level
          ConfigMap learningConfigMap = ConfigMap::fromYamlFile("learning_config.yml");
          if(learningConfigMap.find("Environment") != learningConfigMap.end())
          {
            if(learningConfigMap["Environment"].find("ballThrowTime") != learningConfigMap["Environment"].endMap())
              ballThrowTime = (learningConfigMap["Environment"])["ballThrowTime"];
            if(learningConfigMap["Environment"].find("armHeight") != learningConfigMap["Environment"].endMap())
              armHeight = learningConfigMap["Environment"]["armHeight"];
            if(learningConfigMap["Environment"].find("verbose") != learningConfigMap["Environment"].endMap())
              verbose = (learningConfigMap["Environment"])["verbose"];
          }
        }

        std::string ThrowingEnvironment::getConfigPath()
        {
          // Here we use the environment variable "ROCK_CONFIGURATION_PATH" in
          // order to "find" the smurf file to be loaded. During installation it
          // should be put in "$ROCK_CONFIGURATION_PATH/spacebot_throw_environment".
          std::string configPath = std::string(getenv("ROCK_CONFIGURATION_PATH"));
          if(configPath.empty())
            throw std::runtime_error("WARNING: The ROCK_CONFIGURATION_PATH is not "
                                     "set! Did you \"source env.sh\"?\n");
          return configPath;
        }

        void ThrowingEnvironment::getMotorIDs()
        {
          motorIDs.clear();

          std::vector<mars::interfaces::core_objects_exchange>::iterator it;
          std::vector<mars::interfaces::core_objects_exchange> motorList;
          control->motors->getListMotors(&motorList);

          for(it = motorList.begin(); it != motorList.end(); ++it)
            motorIDs.push_back(it->index);
        }

To reset the environment, we usually have to set the joint angles to the
initial state.

.. code-block:: c++

        void ThrowingEnvironment::resetMARSEnvironment()
        {
          fitness = 0.0;
          evaluation_done = false;
          setStartAngles();
        }

        void ThrowingEnvironment::setStartAngles()
        {
          for(unsigned int i=0; i < motorIDs.size(); i++)
          {
            inputs[i] = startAnglePos(i);
            control->motors->setMotorValue(motorIDs[i], startAnglePos(i));
          }

          dataMutex.lock();
          handleInputValues();
          createOutputValues();
          dataMutex.unlock();
        }

`ThrowingEnvironment::handleMARSError()` will be called when an exception
occurs during simulation. We should set a very bad fitness and finish the
evaluation in the environment.

.. code-block:: c++

        void ThrowingEnvironment::handleMARSError()
        {
          fitness = -DBL_MAX;
          evaluation_done = true;
        }

After each simulation step, this function will be called. Usually we
want to output joint states. We could also output sensor measurements.

.. code-block:: c++

        void ThrowingEnvironment::createOutputValues(void)
        {
          setPositionOfVisualTarget(); // must always be done, falls down otherwise
          outputMotorPositions();
          checkBallPosition();
          checkMaxTime();
        }

        void ThrowingEnvironment::setPositionOfVisualTarget()
        {
          mars::interfaces::NodeId targetId = control->nodes->getID("target_link");
          control->nodes->setPosition(targetId, targetPos);
        }

        void ThrowingEnvironment::outputMotorPositions()
        {
          for(unsigned int i = 0; i < motorIDs.size(); i++)
            outputs[i] = (double)control->motors->getActualPosition(motorIDs[i]);
        }

        void ThrowingEnvironment::checkBallPosition()
        {
          mars::interfaces::NodeId ballId = control->nodes->getID("ball_link");
          mars::utils::Vector ballPos = control->nodes->getPosition(ballId);

          if(ballPos[2] <= -armHeight)
          {
            ballHitX = ballPos[0];
            ballHitY = ballPos[1];
            const double diffX = ballPos[0] - targetPos[0];
            const double diffY = ballPos[1] - targetPos[1];
            const double squaredDist = diffX * diffX + diffY * diffY;
            fitness = -squaredDist;
            evaluation_done = true;
          }
        }

        void ThrowingEnvironment::checkMaxTime()
        {
          if(leftTime > MAX_SIMULATION_TIME) {
            fitness = DBL_MAX;
            evaluation_done = true;
          }
        }

Before a simulation step is computed, we at least should write the motor
commands.

.. code-block:: c++

        void ThrowingEnvironment::handleInputValues()
        {
          setMotorValues();
          checkBallThrowTime();
        }

        void ThrowingEnvironment::setMotorValues()
        {
          for(unsigned int i=0; i < motorIDs.size(); i++)
            control->motors->setMotorValue(motorIDs[i], inputs[i]);
        }

        void ThrowingEnvironment::checkBallThrowTime()
        {
          if(leftTime > ballThrowTime)
            control->joints->removeJoint(control->joints->getID("ball_joint"));
        }

In a contextual environment, we have to set the context on request.

.. code-block:: c++

        double* ThrowingEnvironment::request_context(double *context, int size)
        {
          if(size != 2)
            return NULL;

          targetPos[0] = context[0];
          targetPos[1] = context[1];
          targetPos[2] = -armHeight;

          return targetPos.data();
        }
