# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np
import warnings
from ..utils import from_dict
from ..environment import Environment, ContextualEnvironment
from ..behavior_search import BehaviorSearch
from ..base import Base


class Controller(Base):
    """A controller implements the communication between learning components.

    Controllers organize communication between Environment and BehaviorSearch.
    The code should neither depend on the environment nor on the behavior
    search algorithm so that we can reuse a controller for as many scenarios as
    possible.

    The controller subsection of the configuration dictionary may contain
    the following parameters:

    * n_episodes (int) - number of episodes that will be executed by
      :func:`learn`
    * record_trajectories (bool) - store control signal trajectories (outputs
      of behaviors) of each episode in `self.trajectories_`
    * record_outputs (bool) - store outputs of environment (inputs for
      behaviors) for each episode in `self.outputs_`
    * record_contexts (bool) - store context vectors of each episode in
      `self.contexts_` (only available for contextual environments)
    * n_episodes_before_test (int) - the upper-level policy will be evaluated
      after `n_episodes_before_test` episodes
    * verbose (bool) - print information to stdout

    Parameters
    ----------
    config : dict
        Configuration dictionary for the controller. The environment and the
        behavior search can either be specified in this dictionary or can
        be passed as arguments. In addition, parameters that configurate the
        controller can be passed here in the 'Controller' subsection.

    environment : Environment
        Environment in which we will execute behaviors and learn

    behavior_search : BehaviorSearch, optional (default: None)
        Behavior search that evolves the behavior in the environment

    kwargs : dict
        Additional controller parameters
    """
    def __init__(self, config=None, environment=None, behavior_search=None,
                 **kwargs):
        if config is None:
            config = {}
        else:
            config = from_dict(config)

        if environment is not None:
            self.environment = environment
        elif "Environment" in config:
            self.environment = config["Environment"]
        else:
            raise ValueError("Environment specification is missing.")

        if behavior_search is not None:
            self.behavior_search = behavior_search
        elif "BehaviorSearch" in config:
            self.behavior_search = config["BehaviorSearch"]
        else:
            self.behavior_search = None

        self._check()
        self._init_environment()
        self._init_behavior_search()

        self.inputs = np.zeros(self.n_inputs)
        self.outputs = np.zeros(self.n_outputs)

        self.__dict__.update(kwargs)
        self._set_attribute(config, "n_episodes", 10)
        self._set_attribute(config, "record_trajectories", False)
        self._set_attribute(config, "record_outputs", False)
        self._set_attribute(config, "record_feedbacks", False)
        self._set_attribute(config, "verbose", False)
        self._set_attribute(config, "n_episodes_before_test", None)

        if self.record_trajectories:
            self.trajectories_ = []

        if self.record_outputs:
            self.outputs_ = []

        if self.record_feedbacks:
            self.feedbacks_ = []

        self.episode_cnt = 0

        self.do_test = self.n_episodes_before_test is not None
        if self.do_test:
            self.test_results_ = []

        if self.verbose >= 1:
            print("[Controller] Initialized with")
            print("             - %d inputs" % self.n_inputs)
            print("             - %d outputs" % self.n_outputs)

    def _set_attribute(self, config, name, default):
        value = config.get("Controller", {}).get(name, default)
        # config info available -> value set to config info
        # no config info -> value set to default
        if hasattr(self, name):
            # attribute was set by kwargs
            if value != default:
                # config info *is* available for this attribute
                warnings.warn(
                    "Attribute '%s' exists already as keyword argument "
                    "(value: '%s'). Overwriting with '%s' from configuration "
                    "dictionary." % (name, getattr(self, name), value),
                    stacklevel=2)
                # we warn, but where do we overwrite?
        else:
            # this only happens if the attribute was not defined earlier
            setattr(self, name, value)

    def _init_environment(self):
        self.environment.init()
        self.n_inputs = self.environment.get_num_inputs()
        self.n_outputs = self.environment.get_num_outputs()

    def _init_behavior_search(self):
        if self.behavior_search is not None:
            # Outputs of the environment are inputs for the behavior search
            # and vice versa
            self.behavior_search.init(self.n_outputs, self.n_inputs)

    def _check(self):
        """Check environment and behavior search."""
        if not isinstance(self.environment, Environment):
            raise TypeError("Controller requires subclass of 'Environment'")
        if (self.behavior_search is not None and
                not isinstance(self.behavior_search, BehaviorSearch)):
            raise TypeError("Controller requires subclass of 'BehaviorSearch'")

    def learn(self, meta_parameter_keys=[], meta_parameters=[]):
        """Learn the behavior.

        Parameters
        ----------
        meta_parameter_keys : list
            Meta parameter keys

        meta_parameters : list
            Meta parameter values

        Returns
        -------
        accumulated_feedbacks : array, shape (n_episodes or less,)
            Accumulated feedbacks for each episode. If is_behavior_learning_done is True before the n_episodes is
            reached, the length of accumulated_feedbacks is shorter than n_episodes
        """
        episode_counter = 0
        is_behavior_learning_done = False
        accumulated_feedbacks = []
        while (self.n_episodes > episode_counter) and not is_behavior_learning_done:
            episode_counter += 1
            accumulated_feedbacks.append(self.episode(meta_parameter_keys, meta_parameters))

            is_behavior_learning_done = self.behavior_search.is_behavior_learning_done() or \
                self.environment.is_behavior_learning_done()
        if self.verbose:
            print("is_behavior_learning_done returned these values at termination of the learning process:\n"
                  "behavior_search: %s, environment %s" % (str(self.behavior_search.is_behavior_learning_done()),
                                                           str(self.environment.is_behavior_learning_done())))
        return np.array(accumulated_feedbacks)

    def episode(self, meta_parameter_keys=(), meta_parameters=()):
        """Execute one learning episode.

        Parameters
        ----------
        meta_parameter_keys : array-like, shape = (n_meta_parameters,)
            Meta parameter keys

        meta_parameters : array-like, shape = (n_meta_parameters,)
            Meta parameter values

        Returns
        -------
        accumulated_feedback : float
            Accumulated feedback of the episode
        """
        if self.behavior_search is None:
            raise ValueError("A BehaviorSearch is required to execute an "
                             "episode without specifying a behavior.")

        if self.verbose >= 1:
            print("[Controller] Episode: #%d" % (self.episode_cnt + 1))

        behavior = self.behavior_search.get_next_behavior()
        feedbacks = self.episode_with(behavior, meta_parameter_keys,
                                      meta_parameters)
        self.behavior_search.set_evaluation_feedback(feedbacks)

        accumulated_feedback = np.sum(feedbacks)
        if self.verbose >= 2:
            print("[Controller] Accumulated feedback: %g" % accumulated_feedback)

        self.episode_cnt += 1

        if self.do_test and self.episode_cnt % self.n_episodes_before_test == 0:
            self.test_results_.append(
                self._perform_test(meta_parameter_keys, meta_parameters))

        return accumulated_feedback

    def episode_with(self, behavior, meta_parameter_keys=[],
                     meta_parameters=[], record=True):
        """Execute a behavior in the environment.

        Parameters
        ----------
        behavior : Behavior
            Fix behavior

        meta_parameter_keys : list, optional (default: [])
            Meta parameter keys

        meta_parameters : list, optional (default: [])
            Meta parameter values

        record : bool, optional (default: True)
            Record feedbacks or trajectories if activated

        Returns
        -------
        feedbacks : array, shape (n_steps,)
            Feedback for each step in the environment
        """
        self.environment.reset()
        behavior.set_meta_parameters(meta_parameter_keys, meta_parameters)

        if self.record_trajectories:
            trajectory = []
        if self.record_outputs:
            outputs = []

        while not self.environment.is_evaluation_done():
            # Sense
            self.environment.get_outputs(self.outputs)
            behavior.set_inputs(self.outputs)
            if behavior.can_step():
                behavior.step()
                behavior.get_outputs(self.inputs)
            # Act
            self.environment.set_inputs(self.inputs)
            self.environment.step_action()

            if record:
                if self.record_trajectories:
                    trajectory.append(self.inputs.copy())
                if self.record_outputs:
                    outputs.append(self.outputs.copy())

        feedbacks = self.environment.get_feedback()

        if record:
            if self.record_trajectories:
                self.trajectories_.append(trajectory)
            if self.record_outputs:
                self.outputs_.append(outputs)
            if self.record_feedbacks:
                self.feedbacks_.append(np.sum(feedbacks))
        return feedbacks

    def _perform_test(self, meta_parameter_keys, meta_parameters):
        behavior = self.behavior_search.get_best_behavior()
        performance = np.sum(self.episode_with(
            behavior, meta_parameter_keys, meta_parameters))
        optimum = self.environment.get_maximum_feedback()
        if self.verbose >= 1:
            print("[Controller] Test feedback: %g" % (performance - optimum))
        return performance - optimum


class ContextualController(Controller):
    """Controller for contextual problems.

    See base class "Controller" for details on usage.

    The controller subsection of the configuration dictionary may contain
    the following additional parameters:

    * test_contexts (array-like) - the upper-level policy will be evaluated in
      these contexts
    """
    def __init__(self, config={}, environment=None, behavior_search=None,
                 **kwargs):
        super(ContextualController, self).__init__(
            config, environment, behavior_search, **kwargs)

        self._set_attribute(config, "record_contexts", False)
        self._set_attribute(config, "test_contexts", None)

        if self.record_contexts:
            self.contexts_ = []

        if self.do_test:
            if self.test_contexts is None:
                raise ValueError("You must provide 'test_contexts' if "
                                 "'n_episodes_before_tests' is not None.")

        if self.verbose >= 1:
            print("             - %d context dimensions" % self.n_context_dims)

    def _init_behavior_search(self):
        if self.behavior_search is not None:
            self.n_context_dims = self.environment.get_num_context_dims()
            # Outputs of the environment are inputs for the behavior search
            # and vice versa
            self.behavior_search.init(self.n_outputs, self.n_inputs,
                                      self.n_context_dims)

    def _check(self):
        if not isinstance(self.environment, ContextualEnvironment):
            raise TypeError("ContextualController requires contextual "
                            "environment!")
        if (self.behavior_search is not None and
                isinstance(self.behavior_search, BehaviorSearch)):
            raise TypeError("ContextualController requires contextual "
                            "behavior search!")

    def _negotiate_context(self):
        """Negotiate context."""
        context = self.behavior_search.get_desired_context()
        context = self.environment.request_context(context)
        self.behavior_search.set_context(context)

        if self.record_contexts:
            self.contexts_.append(context)

        return context

    def episode(self, meta_parameter_keys=[], meta_parameters=[]):
        if self.behavior_search is None:
            raise ValueError("A ContextualBehaviorSearch is required to "
                             "execute an episode without specifying a "
                             "behavior.")

        context = self._negotiate_context()

        accumulated_feedback = super(ContextualController, self).episode(
            meta_parameter_keys, meta_parameters)

        if self.verbose >= 2 and context is not None:
            print("[Controller] Context: %s" % context)

        return accumulated_feedback

    def _perform_test(self, meta_parameter_keys, meta_parameters):
        behavior_template = self.behavior_search.get_best_behavior_template()
        results = np.empty(len(self.test_contexts))
        for i, context in enumerate(self.test_contexts):
            actual_context = self.environment.request_context(context)
            if not np.allclose(actual_context, context):
                raise Exception("Evaluation is not possible, could not set "
                                "context.")
            behavior = behavior_template.get_behavior(context)
            # TODO what happens if we do not know the optimum?
            optimum = self.environment.get_maximum_feedback(context)
            current = np.sum(self.episode_with(
                behavior, meta_parameter_keys, meta_parameters, False))
            results[i] = optimum - current
        return results
