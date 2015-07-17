.. quick_start:

===========
Quick Start
===========

We will start with a simple example of how to use an optimizer. We can construct
a CMA-ES optimizer that optimizes two parameters as follows:

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

Take a look at the :ref:`examples-index` to see more complex
use cases.
