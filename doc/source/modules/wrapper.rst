.. _wrapper:

=======
Wrapper
=======

.. currentmodule:: bolero.wrapper

BOLeRo is able to wrap C++ implementations of all components easily with
the wrappers that are provided in this package.

The class :class:`CppBLLoader` provides an interface to load C++ components.
It has the functions

* :func:`~CppBLLoader.acquire_behavior`
* :func:`~CppBLLoader.acquire_behavior_search`
* :func:`~CppBLLoader.acquire_contextual_environment`
* :func:`~CppBLLoader.acquire_environment`
* :func:`~CppBLLoader.acquire_optimizer`

that require a library name as an argument and return a wrapped C++ object.
These libraries have to be registered previously with
:func:`CppBLLoader.load_config_file` which takes the name of a configuration
file as an argument. The configuration file contains the name or location of
one library per line, for example

.. code-block:: text

    pso_optimizer
    mountain_car

C++ components usually need to be configured during acquisition. The
configuration is provided via configuration files. As a convention, the
configuration file `learning_config.yml` is used for this purpose, however,
that depends on the implementation of the component. Another convention is
that, for example an environment only uses the section `Environment`
from the configuration file, an optimizer uses the section `Optimizer`,
etc. For example, the following configuration file configures a
MARS-based environment.

.. code-block:: yaml

    Environment:
        calc_ms: 10
        stepTimeMs: 10
        graphicsUpdateTime: 10
        velocityControl: True
        enableGUI: True

We can load an environment in Python with

.. code-block:: python

    bll = CppBLLoader()
    bll.load_config_file(LIBRARY_CONFIG_FILE)
    env = bll.acquire_environment("mountain_car")
