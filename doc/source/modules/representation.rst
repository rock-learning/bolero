.. _representation:

==============
Representation
==============

Everything that has to be stored not only temporarily in BOLeRo is located
in :mod:`bolero.representation`.

.. currentmodule:: bolero.representation

Behaviors
=========

The purpose of this library is learning behaviors. Behaviors typically
implement the interface :class:`~bolero.representation.Behavior`. It is
important to know that there are some conventions regarding the inputs and
outputs of the generic behavior interface:

* Often get_output and set_input deal with the same kind of data, e.g. a
  :class:`~bolero.representation.DMPBehavior` generates a set of positions,
  velocities and accelerations in Cartesian space or in joint angle space and
  then they expect to get back the same information from a sensor that
  measures the actual positions, velocities and accelerations. However, that
  does not have to be the case for other kind of behaviors.
* If a behavior generates multiple derivatives of the same attribute, e.g.
  positions, velocities and accelerations, it is not per se clear which layout
  would be to best to put these information in a flat vector. We agreed that
  all positions (p), all velocities (v) and all accelerations (a) should be
  stored contiguously, e.g. when the behavior controls 3 joints, the output
  would have the layout :code:`pppvvvaaa`. The reason is that it is easy to
  extract e.g. the position vector with a slice from the output vector.

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

   * - :ref:`Linear <linear_beh>`
     - anything
     - anything
     - linear combination of the inputs
     - weights of the linear mapping

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

   * - :ref:`Sequence of DMPs <dmp_seq>`
     - trajectories in joint space
     - positions, velocities, accelerations
     - positions, velocities, accelerations
     - weights of the internal function approximators and subgoals


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


.. _linear_beh:

Linear Behavior
---------------

A :class:`LinearBehavior` generates a linear mapping :math:`y = W x` from
an input vector :math:`x` (with an additional bias component that is always 1)
to an output vector :math:`y`.


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
A variant of DMPs that works in Cartesian space is
:class:`CartesianDMPBehavior`.

.. include:: ../gen_modules/backreferences/bolero.representation.DMPBehavior.examples
.. raw:: html

    <div style='clear:both'></div>

.. include:: ../gen_modules/backreferences/bolero.representation.CartesianDMPBehavior.examples
.. raw:: html

    <div style='clear:both'></div>


.. _dmp_seq:

Sequence of DMPs
----------------

We can learn a sequence of DMPs. In the class :class:`DMPSequence` allows us to
optimize the DMP weights and the subgoals of the DMPs.
