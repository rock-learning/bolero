.. _mars_environment:

================
MARS Environment
================

This guide will explain how to set up a MARS-based environment for BOLeRo.
We published an example for a MARS environment in a
`separate project <https://github.com/rock-learning/bolero>`_. We will go
through this example step by step here. In the end you will see something
like this:

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

TODO