.. _optimizer:

============
Optimization
============

Optimization plays an important role in behavior learning methods. Optimizers
can be used directly to learn behaviors or they can be used internally by
behavior search algorithms.

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
