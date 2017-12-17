.. _behavior_learning:

=================
Behavior Learning
=================

Behaviors can be learned with many methods from the field of reinforcement
learning, black-box optimization, and evolutionary computation. BOLeRo tries
to provide a set of tools for learning behaviors with several methods and
for implementing new approaches. However, not all behavior learning methods
are compatible to BOLeRo. Our focus is on policy search methods because they
are easy to apply in a robotic application.

The following diagram shows the control flow of learning behaviors with
BOLeRo. An **environment** defines the learning problem. It can execute the
**behavior** and generates the feedback for the **behavior search** algorithm.
The behavior search algorithm generates a new behavior for each episode and
receives the feedback from the environment after each episode. During an
episode (or rollout or trial), the **behavior** is executed in the
**environment**. A behavior measures the state of the environment and
generates an action based on the current state in each step.

.. image:: _static/control_flow.svg
   :width: 80%
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
