.. _py_api_doc:

==========
Python API
==========

This is the class and function reference of the Python API.

You can search for specific modules, classes or functions in the
:ref:`genindex`.

:mod:`bolero.optimizer`: Optimizer
==================================

.. automodule:: bolero.optimizer
    :no-members:
    :no-inherited-members:

Optimizer classes
-----------------
.. currentmodule:: bolero

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   optimizer.ContextualOptimizer
   optimizer.Optimizer
   optimizer.CMAESOptimizer
   optimizer.RestartCMAESOptimizer
   optimizer.IPOPCMAESOptimizer
   optimizer.BIPOPCMAESOptimizer
   optimizer.NoOptimizer
   optimizer.RandomOptimizer

:mod:`bolero.representation`: Representation
============================================

.. automodule:: bolero.representation
    :no-members:
    :no-inherited-members:

Optimizer classes
-----------------
.. currentmodule:: bolero

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   representation.ConstantBehavior
   representation.DummyBehavior
   representation.RandomBehavior

:mod:`bolero.utils`: Utilities
==============================

.. automodule:: bolero.utils
    :no-members:
    :no-inherited-members:

Utility functions
-----------------
.. currentmodule:: bolero

.. autosummary::
   :toctree: modules/generated/
   :template: function.rst

   utils.from_dict
   utils.from_yaml
   utils.log.get_logger
   utils.dependency.compatible_version

Utility classes
---------------
.. currentmodule:: bolero

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   utils.log.HideExtern
