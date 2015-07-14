.. _api_py:

==========
Python API
==========

This is the class and function reference of the Python API.

You can search for specific modules, classes or functions in the
:ref:`genindex`.

:mod:`bolero.environment`: Environment
======================================

.. automodule:: bolero.environment
    :no-members:
    :no-inherited-members:

Environment search classes
--------------------------
.. currentmodule:: bolero

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   environment.Environment
   environment.ContextualEnvironment
   environment.SetContext
   environment.ObjectiveFunction
   environment.ContextualObjectiveFunction
   environment.OptimumTrajectory

:mod:`bolero.behavior_search`: Behavior Search
==============================================

.. automodule:: bolero.behavior_search
    :no-members:
    :no-inherited-members:

Behavior search classes
-----------------------
.. currentmodule:: bolero

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   behavior_search.BehaviorSearch
   behavior_search.ContextualBehaviorSearch
   behavior_search.BlackBoxSearch
   behavior_search.ContextualBlackBoxSearch
   behavior_search.JustOptimizer
   behavior_search.JustContextualOptimizer

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
   optimizer.CREPSOptimizer

:mod:`bolero.representation`: Representation
============================================

.. automodule:: bolero.representation
    :no-members:
    :no-inherited-members:

Behavior classes
----------------
.. currentmodule:: bolero

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   representation.Behavior
   representation.BlackBoxBehavior
   representation.ConstantBehavior
   representation.DummyBehavior
   representation.RandomBehavior
   representation.DMPBehavior

Policy classes
--------------
.. currentmodule:: bolero

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   representation.BehaviorTemplate
   representation.HierarchicalBehaviorTemplate

:mod:`bolero.datasets`: Datasets
================================

.. automodule:: bolero.datasets
    :no-members:
    :no-inherited-members:

Dataset functions
-----------------
.. currentmodule:: bolero

.. autosummary::
   :toctree: modules/generated/
   :template: function.rst

   datasets.make_minimum_jerk

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
