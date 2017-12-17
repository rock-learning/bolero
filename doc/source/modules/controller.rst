.. _controller:

==========
Controller
==========

Controllers manage the interaction between environments, behaviors and
behavior search algorithms.

.. currentmodule:: bolero.controller

The following table gives an overview of the environments that are provided by
BOLeRo.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Controller name
     - Usecase

   * - :ref:`Controller <controller_m>`
     - Behavior search, execute behaviors

   * - :ref:`Contextual Controller <contextual_controller>`
     - Contextual behavior search


.. _controller_m:

Controller
==========

A :class:`Controller` can be used to do behavior search when a behavior search
algorithm is defined or it can be used to execute behaviors in an environment.

.. include:: ../gen_modules/backreferences/bolero.controller.Controller.examples
.. raw:: html

    <div style='clear:both'></div>

.. _contextual_controller:

Contextual Controller
=====================

A :class:`ContextualController` can be used to do contextual behavior search
when a contextual behavior search algorithm is defined.
