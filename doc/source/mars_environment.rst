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

  * :download:`ExampleEnvironment.h <_static/example_environment/src/ExampleEnvironment.h>`
  * :download:`ExampleEnvironment.cpp <_static/example_environment/src/ExampleEnvironment.cpp>`


Building an Environment
=======================

Building a MARS-based environment is straightforward and not much different
to the standard CMake-based build process:

    mkdir build
    cd build
    cmake_debug
    make install

We do not use `cmake ..` because we want to install the environment to our
BOLeRo installation. `cmake_debug` and `cmake_release` do automatically
set the correct `CMAKE_INSTALL_PREFIX`.
