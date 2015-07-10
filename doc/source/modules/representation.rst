.. _representation:

==============
Representation
==============

Everything that has to be stored not only temporarily in BOLeRo is located
in :mod:`bolero.representation`.

.. currentmodule:: bolero.representation

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

   * - :class:`Constant <ConstantBehavior>`
     - anything
     - anything
     - constant
     - none

   * - :class:`Random <RandomBehavior>`
     - anything
     - anything
     - random
     - none

   * - :class:`Dummy <DummyBehavior>`
     - function optimization
     - parameters
     - will be used as output
     - allows direct optimization of the objective function

   * - :class:`Dynamical Movement Primitive <DMPBehavior>`
     - trajectories in joint space or Cartesian space
     - positions, velocities, accelerations
     - positions, velocities, accelerations
     - weights of the internal function approximator