.. _environment:

===========
Environment
===========

Environments define learning problems. They can be used to execute behaviors
and measure their performance.

.. currentmodule:: bolero.environment

Environment
===========

The following table gives an overview of the environments that are provided by
BOLeRo.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Environment name
     - Usecase

   * - :ref:`Objective Function <objective_function>`
     - Benchmark functions for black-box optimization

   * - :ref:`Optimum Trajectory <optimum_trajectory>`
     - Optimize a trajectory to avoid obstacles and minimize the effort


.. _objective_function:

Objective Function
------------------

Several benchmark functions to compare continuous black-box optimizers are
provided by the environment :class:`ObjectiveFunction`. The objective
functions are the same as in the software `COCO
<http://coco.gforge.inria.fr/doku.php>`_. For example, you can see a comparison
of several optimizers on the Rosenbrock function in the following plot.

.. figure:: ../auto_examples/optimizer/images/plot_optimization_001.png
   :target: ../auto_examples/optimizer/plot_optimization.html
   :align: center
   :scale: 50%


.. _optimum_trajectory:

Optimum Trajectory
------------------

The environment :class:`OptimumTrajectory` is designed to use behavior learning
algorithms for simple planning problems. The goal is to avoid obstacles and
minimize the effort used for the trajectory, e.g. by minimizing the velocities
or accelerations. An example for a two-dimensional trajectory is displayed in
the following plot. The obstacles are displayed as red circles on the right
side.

.. figure:: ../auto_examples/behavior_search/images/plot_obstacle_avoidance_001.png
   :target: ../auto_examples/behavior_search/plot_obstacle_avoidance.html
   :align: center
   :scale: 80%


Contextual Environment
======================

The following table gives an overview of the contextual environments that are
provided by BOLeRo.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Environment name
     - Usecase

   * - :ref:`Contextual Objective Function <contextual_objective_function>`
     - Contextual benchmark functions for contextual black-box optimization

   * - :ref:`Catapult <catapult_env>`
     - Benchmark problem for contextual policy search


.. _contextual_objective_function:

Contextual Objective Function
-----------------------------

Several contextual benchmark functions to compare continuous, contextual
black-box optimizers are provided by the environment
:class:`ContextualObjectiveFunction`. The contextual objective functions are
based on the functions that are provided with `COCO
<http://coco.gforge.inria.fr/doku.php>`_.


.. _catapult_env:

Catapult
--------

The catapult environment is a benchmark problem for contextual policy search.
It is a two-dimensional environment like the one displayed in the figur below.

.. figure:: ../auto_examples/environment/images/plot_catapult_001.png
   :target: ../auto_examples/environment/plot_catapult.html
   :align: center
   :scale: 80%

The goal is to hit the ground at a target specified on the x-axis. The target
is given by the context vector.
