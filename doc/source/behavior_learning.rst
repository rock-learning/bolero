.. _behavior_learning:

=================
Behavior Learning
=================

Behaviors can be learned with many methods from the field of reinforcement
learning, black-box optimization and evolutionary computation. BOLeRo tries
to provide the set of tools for learning behaviors with several methods and
for implementing new approaches. However, not all behavior learning methods
are compatible to BOLeRo. Our focus is on policy search methods because they
are easy to apply in a robotic application.

The following diagram shows the control flow of learning behaviors with
BOLeRo. An **environment** defines the learning problem. It can execute the
**behavior** and generates the feedback for the **behavior search** algorithm.
The behavior search algorithm generates a new behavior for each episode and
receives the feedback from the environment after each episode.

.. image:: _static/control_flow.svg
   :alt: Control flow
   :align: center

Black-box optimization can be regarded as a special form of policy search in
which the optimizer does not know anything about the behavior representation.
In this case, the behavior is part of the objective function from the behavior
learning algorithm's perspective.

Other behavior learning methods that are supported by BOLeRo are

* contextual policy search: learns a behavior template that generalizes a
  behavior over a set of contexts or task parameters
* imitation learning: learns from demonstrated behavior

We will now explain the components of BOLeRo in detail.

Behavior
========

The purpose of this library is learning behaviors. Behaviors typically
implement the interface :class:`~bolerorepresentation.Behavior`. It is
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

Optimizer
=========

Optimization algorithms can be used for policy search directly or they can
be used within a behavior search algorithm to optimize parameters internally.
That is the reason why they play an important role in BOLeRo.

Optimizers seek to find some optimum (minimum or maximum) value, e.g. so that

.. math::

    \arg \max_{x \in A} f(x),

where :math:`f` is called objective function. In behavior learning, :math:`f`
is usually a fitness function or a reward function that has to be maximized.
The library provides some black-box optimization algorithms. Black-box in this
case means that no knowledge about the objective is required (e.g. gradients).

Here is a comparison of several optimization algorithms for an unimodal
functions (see :ref:`example_optimizer_plot_optimization.py`
for more details).

.. figure:: auto_examples/optimizer/images/plot_optimization_001.png
   :target: auto_examples/optimizer/plot_optimization.html
   :align: center
   :scale: 50%

Some optimizers are designed for unimodal optimization and some are designed
for multimodal problems (see :ref:`example_optimizer_plot_optimization.py`
for more details). Here is a comparison for these optimizers.

.. figure:: auto_examples/optimizer/images/plot_optimization_002.png
   :target: auto_examples/optimizer/plot_optimization.html
   :align: center
   :scale: 50%
