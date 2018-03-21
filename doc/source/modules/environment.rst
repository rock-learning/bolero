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

   * - :ref:`OpenAI Gym <openaigym>`
     - Benchmark problems for reinforcement learning

   * - :ref:`Optimum Trajectory <optimum_trajectory>`
     - Optimize a trajectory to avoid obstacles and minimize the effort


.. _objective_function:

Objective Function
------------------

Several benchmark functions to compare continuous black-box optimizers are
provided by the environment :class:`ObjectiveFunction`. The objective
functions are the same as in the software `COCO
<http://coco.gforge.inria.fr/doku.php>`_.

.. include:: ../gen_modules/backreferences/bolero.environment.ObjectiveFunction.examples
.. raw:: html

    <div style='clear:both'></div>


.. _openaigym:

OpenAI Gym
----------

The environment :class:`OpenAiGym` is a wrapper for
`OpenAI Gym <https://gym.openai.com>`_ environments.

.. include:: ../gen_modules/backreferences/bolero.environment.OpenAiGym.examples
.. raw:: html

    <div style='clear:both'></div>


.. _optimum_trajectory:

Optimum Trajectory
------------------

The environment :class:`OptimumTrajectory` is designed to use behavior learning
algorithms for simple planning problems. The goal is to avoid obstacles and
minimize the effort used for the trajectory, e.g. by minimizing the velocities
or accelerations.

.. include:: ../gen_modules/backreferences/bolero.environment.OptimumTrajectory.examples
.. raw:: html

    <div style='clear:both'></div>


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

The :class:`Catapult` environment is a benchmark problem for contextual policy search.
It is a two-dimensional environment like the one displayed in the figur below.
The goal is to hit the ground at a target specified on the x-axis. The target
is given by the context vector.

.. include:: ../gen_modules/backreferences/bolero.environment.Catapult.examples
.. raw:: html

    <div style='clear:both'></div>
