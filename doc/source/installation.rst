.. _installation:

============
Installation
============

Prerequisites
=============

BOLeRo is developed for the latest Ubuntu LTS (currently 16.04), has been
tested with 17.04 and the current version of 18.04 and its continuous
integration runs on 14.04.
Using it on Mac OS or Windows is possible, however, the installation might be
more complicated. We recommend to use Docker if you want to take a quick
look at the software. Instructions are
`here <https://github.com/rock-learning/bolero/blob/master/docker/README.md>`_.

In order to install BOLeRo and its dependencies from source, the pybob build
system is used. Therefore, the operating system needs to be prepared according
to the prepare sections in the
`documentation <https://github.com/rock-simulation/pybob>`_.
Afterwards, the following installation instructions can be used.

There are two alternatives to install BOLeRo.

Installation Script
===================

This is the recommended way to install BOLeRo. The installation process for
BOLeRo is a little bit unusual because we depend on the latest versions of
many libraries and support Linux, MacOS and Windows so that we usually have
to download and build them. You can download the installation script and run
it with

.. code-block:: bash

    wget https://raw.githubusercontent.com/rock-learning/bolero/master/bootstrap_bolero.sh
    chmod +x bootstrap_bolero.sh
    ./bootstrap_bolero.sh

Download Source
===============

The latest release of BOLeRo is also available at
`mloss.org <http://mloss.org/software/view/698/>`_ or you can download it from
`Github <https://github.com/rock-learning/bolero/releases>`_. After you unzipped
the release, you can build BOLeRo with the script `install.sh` from the folder
`bolero-dev`. Note that you have to run it from that folder, otherwise the
environment variables will not be configured correctly. The installation
script will also install dependencies.

Environment
===========

The installation script downloads all required packages and installs them in
a target directory that you defined. We call it `$BOLEROPATH`. It has the
following structure:

.. code-block:: text

    $BOLEROPATH
      |-bolero-dev/
      |--install/
      |---bin/
      |---configuration/
      |---include/
      |---lib/
      |---share/
      |--autoproj/
      |---manifest
      |--learning/
      |---bolero/
      |--env.sh

The subdirectory `install` includes all the shared libraries, configurations,
header files, and scripts that you installed. Usually they are not available
in your environment. However, you can source the script `env.sh` with

.. code-block:: text

    source env.sh

to make them available. You could add it to your `.bashrc` to make it permanent.

Optional Packages
=================

The subdirectory autoproj contains a file manifest which includes all activates
packages of BOLeRo. You can activate commented packages by removing "# " in
front of it and calling bob-bootstrap. The manifest file follows the conventions
of `autoproj <http://rock-robotics.org/stable/documentation/autoproj/>`_.

Installing Only the Python Library
==================================

It is possible to install only the Python library BOLeRo without any C++
module via

.. code-block:: bash

    python setup.py install

Testing
=======

To run the Python unit tests, we need nosetests. You can install it with

.. code-block:: bash

    sudo pip install nose

and run it with

.. code-block:: bash

    nosetests bolero -sv

in the bolero main directory `bolero-dev/learning/bolero`.

Building the Documentation
==========================

Install dependencies:

.. code-block:: bash

    sudo apt-get install doxygen
    sudo pip install joblib pillow

Go to the folder 'doc' and run

.. code-block:: bash

    make html

The result will be located in doc/build/html/index.html.
