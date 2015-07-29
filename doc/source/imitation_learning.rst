.. _imiatation_learning:

==================
Imitation Learning
==================

Imitation Learning (IL) is often called Programming by Demonstration (PbD),
Learning from Demonstrations (LfD) or Apprenticeship Learning depending on
the research group. We can distinguish at least two different types of
imitation learning:

* kinesthetic teaching: a person moves the joints of a robot and the motion
  will be recorded from the robot directly
* learning from recorded movements: a person moves and the motion is recorded
  via some kind of motion capturing system

The latter is usually much more difficult since it is not always possible to
reproduce the behavior with another agent because of different kinematic
structures.

A basic overview of the field can be found in [Billard2013]_.

Imitation with Dynamical Movement Primitives
--------------------------------------------

.. currentmodule:: bolero.representation

:class:`~DMPBehavior` s are arbitrarily parametrizable trajectories that adapt
well to changing meta parameters like goal, or temporal scaling. With a
given demonstration that consists of :math:`T` positions, velocities, and
accelerations

.. math::

    (\boldsymbol{x}_1, \dot{\boldsymbol{x}}_1, \ddot{\boldsymbol{x}}_1, \boldsymbol{x}_2, \dot{\boldsymbol{x}}_2, \ddot{\boldsymbol{x}}_2, \ldots)

we can determine the weights of the DMP so that we reproduce the shape of the
demonstration.

The movement dynamics of a DMP is determined by the transformation system.
It generates a goal directed movement based on the current position, velocity
and acceleration and incorporates a learnable, time-dependent term called
forcing term. The parameters of the forcing term can be changed to mimic a
demonstrated movement (imitation learning) or to maximize a given feedback or
reward function (reinforcement learning). However, the influence of the forcing
term decays as the time approaches the maximum execution time of the trajectory.
Hence, the generated movement is guaranteed to reach the goal
:math:`\boldsymbol{g}`.

Let us take a closer look at imitation learning with DMPs: the generated
acceleration :math:`\ddot{\boldsymbol{x}}` at time step :math:`t` in this
system is similar to

.. math::

	\ddot{x}_t = \frac{K (g - x_{t-1})- D \tau \dot{x}_{t-1} - K (g - x_0) s + K f(s)}{\tau^2},

where :math:`g` is the desired goal, :math:`\boldsymbol{x}` is the position,
:math:`\boldsymbol{x}` is the velocity, :math:`s` is the phase variable
:math:`\boldsymbol{x}_0` is the start, :math:`f(s)` is the forcing term and
:math:`\tau` is the duration of the trajectory. :math:`K` and :math:`D` are
constants. In order to follow a demonstrated trajectory we compute the required
forcing terms by solving for :math:`f(s)`:

.. math::

	\frac{\tau^2 \ddot{\boldsymbol{x}}_t - K (\boldsymbol{g} - \boldsymbol{x}_{t-1}) - D \tau \dot{\boldsymbol{x}}_{t-1} - K (\boldsymbol{g} - \boldsymbol{x}_0) s}{K} = f_\text{target}(s).

These are targets for the function approximator that generates :math:`f`.
Now, we just need to apply any supervised learning algorithm that minimizes the
error :math:`E` of the actual targets :math:`f(s)` with respect to the desired
targets :math:`f_\text{target}(s)`

.. math::

    E = \sum_s (f_\text{target}(s) - f(s))^2.

Suppose we have :math:`N` demonstrations stored in an array X. We can use the
imitation learning algorithm with only a few lines of code:

.. doctest::

  >>> import numpy as np
  >>> from bolero.datasets import make_minimum_jerk
  >>> from bolero.representation import DMPBehavior
  >>> dmp = DMPBehavior()
  >>> dmp.init(6, 6)
  >>> dmp.imitate(*make_minimum_jerk(np.zeros(2), np.ones(2), 1.0, 0.01))
  >>> dmp
  DMPBehavior(...)


Literature
----------

.. [Billard2013] Billard, Aude; Grollman; Daniel;
        `Robot learning by demonstration
        <http://www.scholarpedia.org/article/Robot_learning_by_demonstration>`_,
        2013, Scholarpedia 8(12):3824.
