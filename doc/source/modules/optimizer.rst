.. _optimizer:

============
Optimization
============

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

.. currentmodule:: bolero.optimizer

Optimizers
==========

The following table gives an overview of the optimizers that are provided by
BOLeRo.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Optimizer name
     - Usecase

   * - :ref:`NoOptimizer <no_opt>`
     - anything

   * - :ref:`RandomOptimizer <random_opt>`
     - anything

   * - :ref:`CMAESOptimizer <cmaes_opt>`
     - 5 - 2000 parameters, ill-conditioned, non-separable, unimodal objective
       functions

   * - :ref:`IPOPCMAESOptimizer <ipop_cmaes_opt>`
     - 5 - 2000 parameters, ill-conditioned, non-separable, multimodal
       objective functions

   * - :ref:`BIPOPCMAESOptimizer <bipop_cmaes_opt>`
     - 5 - 2000 parameters, ill-conditioned, non-separable, multimodal
       objective functions


Here is a comparison of several optimization algorithms for an unimodal
functions (see :ref:`example_optimizer_plot_optimization.py`
for more details).

.. figure:: ../auto_examples/optimizer/images/plot_optimization_001.png
   :target: ../auto_examples/optimizer/plot_optimization.html
   :align: center
   :scale: 50%

Some optimizers are designed for unimodal optimization and some are designed
for multimodal problems (see :ref:`example_optimizer_plot_optimization.py`
for more details). Here is a comparison for these optimizers.

.. figure:: ../auto_examples/optimizer/images/plot_optimization_002.png
   :target: ../auto_examples/optimizer/plot_optimization.html
   :align: center
   :scale: 50%


.. _no_opt:

No Optimizer
------------

A :class:`NoOptimizer` does not optimize at all. It can be used as a baseline
for optimizers.


.. _random_opt:

Random Optimizer
----------------

A :class:`RandomOptimizer` explores the parameter space randomly. It can be
used as a baseline for optimizers.


.. _cmaes_opt:

CMA-ES
------

:class:`CMAESOptimizer` implements the popular black-box optimizer
`Covariance Matrix Adaption Evolution Strategy
<https://en.wikipedia.org/wiki/CMA-ES>`_. It is a local optimizer although
it can be used to search for global optima with a large population size.
However, CMA-ES with a restart strategy like :ref:`IPOPCMAESOptimizer
<ipop_cmaes_opt>` or :ref:`BIPOPCMAESOptimizer <bipop_cmaes_opt>` might work
better in this case.

The complexity of CMA-ES depends on the number of parameters :math:`N` and is
cubic :math:`O(N^3)` because we approximate the covariance matrix of the
parameter vector and compute its singular value decomposition. Hence, it
should not be used with too many parameters.

CMA-ES represents its current solution by a Gaussian search distribution with
mean :math:`\mu` and covariance :math:`\sigma \boldsymbol{C}` internally. During
one iteration of the algorithm we will draw several samples from the
distribution and compute their fitness. CMA-ES will order the samples and
compute their weights for the next update based on their rank. That is the
reason why CMA-ES is robust to rank-preserving transformations of the objective
function.

The optimization procedure is displayed in the following figure. Each image
displays one generation of samples. Each generation is sampled from the same
search distribution. The old search distribution is displayed by an orange
equiprobable ellipse and the updated search distribution is displayed by a
green ellipse.

.. figure:: ../auto_examples/optimizer/images/plot_cmaes_002.png
   :target: ../auto_examples/optimizer/plot_cmaes.html
   :align: center
   :scale: 50%


.. _ipop_cmaes_opt:

IPOP-CMA-ES
-----------

CMA-ES is a local optimizer. It cannot escape local minima. Hence, to turn it
into a global optimizer, we have to restart the optimization process when it
gets stuck until we find a global optimum. One restart strategy is implemented
by :class:`IPOPCMAESOptimizer`. IPOP-CMA-ES doubles the population size after
each restart.


.. _bipop_cmaes_opt:

BIPOP-CMA-ES
------------

Another restart strategy is implemented by :class:`BIPOPCMAESOptimizer`.
BIPOP-CMA-ES can decide whether it makes more sense to decrease or increase
the population size.


Contextual Optimizers
=====================

The following table gives an overview of the contextual optimizers that are
provided by BOLeRo.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Optimizer name
     - Usecase

   * - :ref:`CREPSOptimizer <creps_opt>`
     - 5 - 500 parameters and 1 - 5 context dimensions


.. _creps_opt:

C-REPS Optimizer
----------------

The :class:`CREPSOptimizer` implements the algorithm Contextual Relative
Entropy Policy Search which is a state of the art approach for contextual
optimization.
