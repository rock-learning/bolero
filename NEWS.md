# Release History

## Version 1.1

Not released yet.

### Features

* New representation: ProMPBehavior

### Bugfixes

* DMPBehavior: Use pseudo inverse in imitation learning for DMPs. Imitating
  trajectories with less points than DMP weights is now possible.
* DMPBehavior: Fix issue with many decimal places in dt.
* Optimizer: Is now an abstract base class as it should have been.
* REPSOptimizer / CREPSOptimizer: Use analytical gradient for optimization.
  This improves the computational efficiency and gives slightly better
  results.

## Version 1.0

2018/02/11

### Maintenance

* Make BOLeRo compatible to latest version of OpenAI Gym (0.9.6)

### Breaking Changes

* C++: BehaviorSearch, Optimizer, and Environment have an additional
  parameter 'config' in their 'init' function. The YAML-based configuration
  string will be given. Each implementation of this function must be able to
  cope with an empty config string. In this case all parameters will be
  set to their default values.
* C++: The controller will run 'reset' of the environment once directly after
  'init' before the first episode is executed.
* C++: Controller only passes corresonding sections of the learning
  configuration to the components.
* pso_optimizer: uses config section 'Optimizer' (previously:
  'BehaviorSearch Parameters').

## Version 0.2

2018/01/23

### Features

* New ContextualOptimizer: C-CMA-ES (based on CMA-ES)
* Support for Windows and MacOS

### Documentation

* Documented `context_features` of CREPSOptimizer

## Version 0.1

2017/12/19

### Features

* Continuous integration with Travis CI and CircleCI
* Added docker image
* New behavior search: Monte Carlo RL
* New optimizer: relative entropy policy search (REPS)
* New optimizer: ACM-ES (CMA-ES with surrogate model)

### Bugfixes

* DMPSequence works with multiple dimensions
* Minor fixes in docstrings
* Multiple minor fixes for Travis CI
* Fixed scaling issues in C-REPS

### Documentation

* Documented merge policy
* Added meta information about the project to the manifest.xml
* Updated documentation on how to build custom MARS environments

## Version 0.0.1

2017/05/19

First public release.

### Breaking Changes

In comparison to the old behavior learning framework used by the DFKI RIC and
the University of Bremen, we changed the following details:

* Python interface: changed signature int `Environment.get_feedback(np.ndarray)`
  to `np.ndarray Environment.get_feedback()`
* Python interface: `ContextualEnvironment` is now a subclass of `Environment`
* Python interface: renamed `Environment.get_maximal_feedback` to
  `Environment.get_maximum_feedback`
* Python and C++ interface: `Behavior` constructor does not take any arguments,
  instead the function `Behavior.init(int, int)` has been introduced to
  determine the number of inputs and outputs and initialize the behavior
* Python interface: Optimizer and ContextualOptimizer are independent
* Python interface: BehaviorSearch and ContextualBehaviorSearch are independent
