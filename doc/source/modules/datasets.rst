.. _datasets:

========
Datasets
========

BOLeRo provides datasets, e.g. for imitation learning.

.. currentmodule:: bolero.datasets

The following table gives an overview of the datasets that are provided by
BOLeRo.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Dataset name
     - Usecase

   * - :ref:`Minimum jerk <minimum_jerk>`
     - Minimum jerk trajectory


.. _minimum_jerk:

Minimum Jerk
============

:func:`make_minimum_jerk` provides a trajectory that minimizes the jerk, i.e.
the 3rd derivative of the position with respect to time. An example is
shown in the following plot. The trajectory goes from 0 to 1 in one dimension
and the position, velocity and acceleration of the trajectory are displayed.

.. figure:: ../auto_examples/datasets/images/plot_minimum_jerk_001.png
   :target: ../auto_examples/datasets/plot_minimum_jerk.html
   :align: center
   :scale: 50%
