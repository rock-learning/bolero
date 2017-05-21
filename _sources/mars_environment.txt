.. _mars_environment:

================
MARS Environment
================

This guide will explain how to set up a MARS-based environment for BOLeRo.


Folder and File Structure of an Environment
===========================================

The structure of a project that provides a MARS-based environment always looks
very similar to this one. You can download files from an example project by
clicking on the links.

**example_environment/**

* :download:`README.md <_static/example_environment/README.md>` - provides
  information for the user
* :download:`CMakeLists.txt <_static/example_environment/CMakeLists.txt>`
  - defines how we can build this project with `CMake <http://www.cmake.org/>`_
* :download:`manifest.xml <_static/example_environment/manifest.xml>` -
  defines dependencies for `Rock <http://rock-robotics.org>`_ or `ROS
  <http://www.ros.org/>`_
* :download:`example_environment.pc.in
  <_static/example_environment/example_environment.pc.in>` - defines package
  information like the location of the library, compiler flags etc.
* **configuration/** - contains MARS scenes and config files which are used by
  the environemnt

  * :download:`example.scn
    <_static/example_environment/configuration/example.scn>`
  * :download:`example_vc.scn
    <_static/example_environment/configuration/example_vc.scn>`

* **src/** - contains source files for the environment implementation

  * :download:`ExampleEnvironment.h
    <_static/example_environment/src/ExampleEnvironment.h>`
  * :download:`ExampleEnvironment.cpp
    <_static/example_environment/src/ExampleEnvironment.cpp>`


Building an Environment
=======================

Building a MARS-based environment is straightforward and not much different
to the standard CMake-based build process:


.. code-block:: bash

    mkdir build
    cd build
    cmake_debug
    make install

We do not use `cmake ..` because we want to install the environment to our
BOLeRo installation. `cmake_debug` and `cmake_release` do automatically
set the correct `CMAKE_INSTALL_PREFIX`.


Creating Your Own Environment
=============================

Starting from the example environment, to create a new simple environment you
have to do the following steps.

Changes in the folder names and filenames
-----------------------------------------

  * copy the example folder and rename it
  * delete all scene files in the configuration folder and put the new ones in
    it (the names of the new scene don't have to be like the environment name
    but it would make things easier)
  * rename the :download:`example_environment.pc.in
    <_static/example_environment/example_environment.pc.in>` and put your
    environment name in front
  * go in the **src/** folder and rename the source files
  * go back to the main folder and edit the :download:`CMakeLists.txt
    <_static/example_environment/CMakeLists.txt>`

    * change the project name to your environment name
    * go to the section where the source files are set and correct the names to
      the right ones
    * go to the end of the file where the scene files are installed and change
      the name of the files to your scene files

Changes in the source
---------------------

  * :download:`src/ExampleEnvironment.h
    <_static/example_environment/src/ExampleEnvironment.h>`:
    In the header file you search for every spot with the word *example* and
    exchange it with the name of your environment.

  * :download:`src/ExampleEnvironment.cpp
    <_static/example_environment/src/ExampleEnvironment.cpp>`:
    First do the same as in the header file (exchange alle *example* with
    your environment name).

Methods
-------

In the following we give a brief description of the methods and what changes
have to be done.

**constructor**:

  *description*:

    * Constructor method of your environment class
    * calls the parent constructor (Environment and MARSEnvironment)
    * initialize constants:

      * MAX_TIME     : maximum time after which the evaluation is considerd as done
      * numJoints    : the number of joints which are used by the environment
      * numAllJoints : the total number of joint of the scene

  *changes*:

    * change *numJoints* and *numAllJoints* to the right values

**initMARSEnvironment()**:

  *description*:

    * Initialise you environment:
    * looks in the *learning_config.yml* wether the velocity or position controlled scene should be loaded
    * loads the scene in MARS
    
  *changes*:

    * change the name of the normal scene and the velocity controlled scene to your scene file name.

**resetMARSEnvironment()**

  *description*:

    * resets the environment to the start state

  *changes*: None

**handleMARSError()**

  *description*:

    * Set the evaluation_done variable to *True* and the fitness to "DBL_MAX"

  *changes*: None

**getSensorIDs()**

  *description*:

    * This Method should search for the id's of the motors you want to use in the environment.
    * The id's are saved in a vector called *sensorIDs*.

  *changes*:

    * In this example the method seaches for the id's of the sensors with the names: "Motor_Angles", "Endeffector_position", "Endeffector_rotation", "Endeffector_velocity".
    * You have to change these names with the names of the sensors in your scene.

**getMotorIDs()**

  *description*:

    * This Method should search for the id's of the motors you want to use in the environment.
    * The id's are saved in a vector called *motorIDs*.

  *changes*:

    * In this example we go through all motors and save their id's in the vector.
    * If you dont want all of your motors to be changed by the environment you have to change this here.

**getNumInputs()**

  *description*:

    * This Method returns the number of inputs this environment can handle

  *changes*:

    * Change the return value to the number of motors you want to control with your environment

**getNumOutputs()**

  *description*:

    * This Method returns the number of outputs your environment creates

  *changes*:

    * Change the return to the number of outputs you want to give

**createOutputValues()**

  *description*:

    * In this method we create the output values and fill them into the array *outputs*.
    * The *outputs* array is part of the mars_environment class. The length of this array depends on the number your *getNumOutputs()* function returns.

  *changes*:

  * You have to go through your sensorID's you got from *getSensorIDs()*, read the sensor data from each sensor and write them into the *outputs* array

**handleInputValues()**

  *description*:

    * This method is called with every step and should handle the new inputs.
    * The input data is in the *inputs* array which is also part of the mars_environment class.
    * The length of this array depends on the number your *getNumInputs()* function returns.

  *changes*:

    * In this method we simply go through the *inputs* array and pass the values to the motors of the loaded scene.
    * If you want to do something different with the inputs you ahve to do this here.

**isEvaluationDone()**

  *description*:

    * *True* if the MAX_TIME is reached or an event sets the evaluation_done to *True*

  *changes*: None

**getFeedback(double *feedback)**

  *description*:

    * returns the feedback generated by the evaluation

  *changes*: None
