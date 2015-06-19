# News

In comparison to the old behavior learning framework used by the DFKI RIC and
the University of Bremen, we changed the following details:

* Python interface: changed signature int `Environment.get_feedback(np.ndarray)`
  to `np.ndarray Environment.get_feedback()`
* Python interface: `ContextualEnvironment` is now a subclass of `Environment`
* Python interface: renamed `Environment.get_maximal_feedback` to
  `Environment.get_maximum_feedback`
