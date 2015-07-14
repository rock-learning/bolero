.. _representation:

==============
Representation
==============

Everything that has to be stored not only temporarily in BOLeRo is located
in :mod:`bolero.representation`.

.. currentmodule:: bolero.representation

Behaviors
=========

The following table gives an overview of the behaviors that are provided by
BOLeRo.

.. list-table::
   :header-rows: 1
   :widths: 25 25 15 15 20

   * - Behavior name
     - Usecase
     - Inputs
     - Outputs
     - Parameters

   * - :ref:`Constant <constant_beh>`
     - anything
     - anything
     - constant
     - none

   * - :ref:`Random <random_beh>`
     - anything
     - anything
     - random
     - none

   * - :ref:`Dummy <dummy_beh>`
     - function optimization
     - parameters
     - will be used as output
     - allows direct optimization of the objective function

   * - :ref:`Dynamical Movement Primitive <dmp_beh>`
     - trajectories in joint space or Cartesian space
     - positions, velocities, accelerations
     - positions, velocities, accelerations
     - weights of the internal function approximator

.. _constant_beh:

Constant Behavior
-----------------

A :class:`ConstantBehavior` always produces a constant output that cannot
be changed. It can be used as a behavior baseline.

.. _random_beh:

Random Behavior
---------------

A :class:`RandomBehavior` always produces a random output that is completely
random and normal distributed. It can be used as a behavior baseline.

.. _dummy_beh:

Dummy Behavior
--------------

A :class:`DummyBehavior` always produces the output that has been given as
parameters from the optimizer. It can be used in cases where no behavior is
required actually, e.g. for plain function optimization or where the behavior
is encoded in the environment.

.. _dmp_beh:

Dynamical Movement Primitive
----------------------------

Dynamical movement primitives represent trajectories :class:`DMPBehavior`, e.g.
in joint space. They can generalize over several meta-parameters (goal,
velocity at the goal, execution time) and can be learned from demonstrations.

.. figure:: ../auto_examples/representation/images/plot_dmp_meta_parameters_001.png
   :target: ../auto_examples/representation/plot_dmp_meta_parameters.html
   :align: center
