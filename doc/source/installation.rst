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

Testing
=======

To run the Python unit tests, we need nosetests. You can install it with

.. code-block:: bash

    sudo pip install nose

and run it with

.. code-block:: bash

    nosetests

in the bolero main directory.
