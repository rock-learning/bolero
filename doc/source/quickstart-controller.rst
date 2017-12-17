.. quickstart-controller:

===============================
Quick Start with the Controller
===============================

In this section, we will give you a brief introduction how to use the C++
implementation of a controller to run experiments with BOLeRo.

We call the application that runs experiments in BOLeRo "controller". It reads
configuration files and connects learning algorithms and problems.
An experiment in BOLeRo can be specified with only one YAML file.

You can run the controller with

.. code-block:: bash

    bolero_controller

You can configure it with two environment variables:

* BL_LOG_PATH - the path in which results will be logged, default: "."
* BL_CONF_PATH - the path from which we load the configuration file, default: "."

The configuration file is called learning_config.yml. It will have the
following form:

.. code-block:: yaml

    Environment:
        type: ...
        ...
    BehaviorSearch:
        type: ...
        ...
    Controller:
        MaxEvaluations: 1000
        LogAllBehaviors: false
        GenerateFitnessLog: true
        LogResults: false
        EvaluateExperiment: false
        TestEveryXRun: 0

Results will be stored in a directory that you specified. That includes
fitness values and intermediate results.