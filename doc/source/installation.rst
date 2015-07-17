.. _installation:

============
Installation
============

The installation process for BOLeRo is a little bit unusual because we depend
on the latest versions of many libraries and support Linux, MaxOS and Windows
so that we usually have to download and build them. You can download the
installation script and run it with

.. code-block:: bash

    git clone git@git.hb.dfki.de:team-learning/bootstrap.git --branch bolero
    cd bootstrap
    ./bootstrap.sh

Confirm everything and follow the instructions.

Environment
===========

The installation script downloads all required packages and installs them in
a target directory that you defined. We call it `$BOLEROPATH`. It has the
following structure:

.. code-block:: text

    $BOLEROPATH
      |-bolero/
      |-install/
      |--bin/
      |--configuration/
      |--include/
      |--lib/
      |--share/
      |-...
      |-env.sh

The subdirectory `install` includes all the shared libraries, configurations,
header files, and scripts that you installed. Usually they are not available
in your environment. However, you can source the script `env.sh` with

.. code-block:: text

    source env.sh

to make them available. You could add it to your `.bashrc` to make it permanent.

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

    nosetests

in the bolero main directory.
