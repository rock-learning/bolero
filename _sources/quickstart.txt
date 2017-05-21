.. quick_start:

=======================
Quick Start with Python
=======================

In this section, we will give you a brief introduction how to use the Python
interface of BOLeRo.

Using a Controller
==================

A :ref:`Controller <controller>` provides a simple tool to organize the whole
behavior learning process. Let us start with a simple example. We need an
environment that specifies a learning problem. Let's say, we want to avoid
obstacles in a two-dimensional space and minimize the accelerations at the
same time.

.. doctest::

    >>> from bolero.environment import OptimumTrajectory
    >>> import numpy as np
    >>> n_task_dims = 2
    >>> obstacles = [np.array([0.5, 0.5]), np.array([0.6, 0.8]), np.array([0.8, 0.6])]
    >>> x0, g = np.zeros(n_task_dims), np.ones(n_task_dims)
    >>> execution_time, dt = 1.0, 0.01
    >>> env = OptimumTrajectory(
    ...     x0, g, execution_time, dt, obstacles, penalty_goal_dist=1.0,
    ...     penalty_obstacle=1000.0, penalty_acc=1.0)
    >>> env
    OptimumTrajectory(...)

In order to learn a behavior, we need a behavior, e.g. a DMP that defines an
end-effector trajectory.

.. doctest::

    >>> from bolero.representation import DMPBehavior
    >>> beh = DMPBehavior(execution_time, dt, n_features=10)
    >>> beh
    DMPBehavior(...dt=0.01, execution_time=1.0, n_features=10)

Next, we have to specify how we want to learn the behavior. We will use a
black-box optimizer to optimize the parameters of the DMP. We have to combine
the behavior and the optimizer to a behavior search component.

.. doctest::

    >>> from bolero.behavior_search import BlackBoxSearch
    >>> from bolero.optimizer import CMAESOptimizer
    >>> bs = BlackBoxSearch(beh, CMAESOptimizer(variance=100.0 ** 2, random_state=0))
    >>> bs
    BlackBoxSearch(behavior=DMPBehavior(...), ..., optimizer=CMAESOptimizer(...))

Now, we have an environment that defines our problem and a behavior search
component that is able to find a solution. We can use a controller to
organize the behavior search.

.. doctest::

    >>> from bolero.controller import Controller
    >>> controller = Controller(
    ...     environment=env, behavior_search=bs, n_episodes=30)
    >>> controller
    Controller(...)
    >>> feedbacks = controller.learn(["x0", "g"], [x0, g])
    >>> feedbacks
    array([-2512... -875...])

Optimization
============

We can use components from BOLeRo without using all the other parts. We will
demonstrate that in this example. We can construct a CMA-ES optimizer that
optimizes two parameters as follows:

.. doctest::

    >>> from bolero.optimizer import CMAESOptimizer
    >>> opt = CMAESOptimizer(random_state=0)
    >>> opt
    CMAESOptimizer(active=False, ..., initial_params=None, ..., variance=1.0)
    >>> opt.init(2)

and generate a new parameter vector with

.. doctest::

    >>> import numpy as np
    >>> params = np.empty(2)
    >>> opt.get_next_parameters(params)
    >>> params
    array([ 1.7...,  0.4...])

Now we can compute the feedback and give it back to the optimizer. Then we can
check what are the best parameters so far.

.. doctest::

    >>> feedback = (params - np.array([0.3, -0.7])) ** 2
    >>> opt.set_evaluation_feedback(feedback)
    >>> opt_params = opt.get_best_parameters()
    >>> opt_params
    array([ 1.7...,  0.4...])

Examples
========

Take a look at the :ref:`examples-index` to see more complex
use cases.
