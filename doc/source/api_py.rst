.. _api_py:

==========
Python API
==========

This is the class and function reference of the Python API.

You can search for specific modules, classes or functions in the
:ref:`genindex`.

:mod:`bolero.controller`: Controller
====================================

.. automodule:: bolero.controller
    :no-members:
    :no-inherited-members:

Controller classes
------------------
.. currentmodule:: bolero.controller

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   Controller
   ContextualController

:mod:`bolero.environment`: Environment
======================================

.. automodule:: bolero.environment
    :no-members:
    :no-inherited-members:

Environment search classes
--------------------------
.. currentmodule:: bolero.environment

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   Environment
   ContextualEnvironment
   SetContext
   ObjectiveFunction
   ContextualObjectiveFunction
   OptimumTrajectory
   Catapult
   OpenAiGym

:mod:`bolero.behavior_search`: Behavior Search
==============================================

.. automodule:: bolero.behavior_search
    :no-members:
    :no-inherited-members:

Behavior search classes
-----------------------
.. currentmodule:: bolero.behavior_search

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   BehaviorSearch
   ContextualBehaviorSearch
   BlackBoxSearch
   ContextualBlackBoxSearch
   JustOptimizer
   JustContextualOptimizer
   MonteCarloRL

:mod:`bolero.optimizer`: Optimizer
==================================

.. automodule:: bolero.optimizer
    :no-members:
    :no-inherited-members:

Optimizer classes
-----------------
.. currentmodule:: bolero.optimizer

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   Optimizer
   ContextualOptimizer
   CMAESOptimizer
   RestartCMAESOptimizer
   IPOPCMAESOptimizer
   BIPOPCMAESOptimizer
   NoOptimizer
   RandomOptimizer
   REPSOptimizer
   CREPSOptimizer
   SkOptOptimizer
   ACMESOptimizer

:mod:`bolero.representation`: Representation
============================================

.. automodule:: bolero.representation
    :no-members:
    :no-inherited-members:

Behavior classes
----------------
.. currentmodule:: bolero.representation

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   Behavior
   BlackBoxBehavior
   ConstantBehavior
   DummyBehavior
   RandomBehavior
   DMPBehavior
   CartesianDMPBehavior
   DMPSequence
   LinearBehavior

Policy classes
--------------
.. currentmodule:: bolero.representation

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   BehaviorTemplate
   HierarchicalBehaviorTemplate

:mod:`bolero.datasets`: Datasets
================================

.. automodule:: bolero.datasets
    :no-members:
    :no-inherited-members:

Dataset functions
-----------------
.. currentmodule:: bolero.datasets

.. autosummary::
   :toctree: modules/generated/
   :template: function.rst

   make_minimum_jerk

:mod:`bolero.wrapper`: Wrapper
==============================

.. automodule:: bolero.wrapper
    :no-members:
    :no-inherited-members:

Wrapper classes
---------------
.. currentmodule:: bolero.wrapper

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   CppBLLoader

:mod:`bolero.utils`: Utilities
==============================

.. automodule:: bolero.utils
    :no-members:
    :no-inherited-members:

Utility functions
-----------------
.. currentmodule:: bolero.utils

.. autosummary::
   :toctree: modules/generated/
   :template: function.rst

   from_yaml
   from_yaml_string
   from_dict
   log.get_logger
   dependency.compatible_version

Utility classes
---------------
.. currentmodule:: bolero.utils

.. autosummary::
   :toctree: modules/generated/
   :template: class.rst

   log.HideExtern
   ranking_svm.RankingSVM
