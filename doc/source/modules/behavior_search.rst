.. _behavior_search:

===============
Behavior Search
===============

Behavior search algorithms provide new behaviors that can be tested in the
environments and learn from the feedback.

.. currentmodule:: bolero.behavior_search

Behavior Search
===============

The following table gives an overview of the behavior search methods that are
provided by BOLeRo.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Behavior search name
     - Usecase

   * - :ref:`Black-box Search <black_box_search>`
     - Policy search, behavior is considered to be a black box


.. _black_box_search:

Black-box Search
----------------

A :class:`BlackBoxSearch` combines an :class:`~bolero.optimizer.Optimizer` and a
:class:`~bolero.representation.BlackBoxBehavior` for `direct policy search
<https://en.wikipedia.org/wiki/Reinforcement_learning#Direct_policy_search>`_.
The optimizer does not need to know anything about the behavior except its
number of parameters and the performance in the environment to do direct policy
search.


Contextual Behavior Search
==========================

The following table gives an overview of the contextual behavior search methods
that are provided by BOLeRo.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Behavior search name
     - Usecase

   * - :ref:`ContextualBlackBoxSearch <contextual_black_box_search>`
     - Contextual policy search, behavior is considered to be a black box


.. _contextual_black_box_search:

Contextual Black-box Search
---------------------------

A :class:`ContextualBlackBoxSearch` combines a
:class:`~bolero.optimizer.ContextualOptimizer` and a
:class:`~bolero.representation.BlackBoxBehavior` for contextual policy search.
The contextual optimizer does not need to know anything about the behavior
except its number of parameters and the performance in the environment to do
contextual policy search.
