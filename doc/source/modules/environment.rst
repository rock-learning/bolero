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

   * - Optimizer name
     - Usecase

   * - :ref:`Objective Function <objective_function>`
     - Benchmark functions for black-box optimization

   * - :ref:`Optimum Trajectory <optimum_trajectory>`
     - Optimize a trajectory to avoid obstacles and minimize the effort


.. _objective_function:

Objective Function
------------------

:class:`ObjectiveFunction`


.. _optimum_trajectory:

Optimum Trajectory
------------------

:class:`OptimumTrajectory`


Contextual Environment
======================

The following table gives an overview of the contextual environments that are
provided by BOLeRo.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Optimizer name
     - Usecase

   * - :ref:`Contextual Objective Function <contextual_objective_function>`
     - Contextual benchmark functions for contextual black-box optimization


.. _contextual_objective_function:

Contextual Objective Function
-----------------------------

:class:`ContextualObjectiveFunction`
