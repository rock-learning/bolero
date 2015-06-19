# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>

import numpy as np
from .behavior_search import BehaviorSearch, ContextualBehaviorSearch
from .behavior_search import PickableMixin, FixableMixin
from ..optimizer import Optimizer, ContextualOptimizer
from ..representation import BlackBoxBehavior, DummyBehavior
from ..utils.module_loader import from_dict


class BlackBoxSearchMixin(object):
    """Mixin for black-box behavior search.

    This mixin can be used in both contextual and non-contextual scenarios.
    """
    def __init__(self, behavior, optimizer, metaparameter_keys=[],
                 metaparameter_values=[]):
        self.behavior = behavior
        self.optimizer = optimizer
        self.metaparameter_keys = metaparameter_keys
        self.metaparameter_values = metaparameter_values

    def _init_helper(self, n_inputs, n_outputs):
        # Initialize behavior
        if isinstance(self.behavior, dict):
            self.behavior["num_inputs"] = n_inputs
            self.behavior["num_outputs"] = n_outputs
            self.behavior = from_dict(self.behavior)
        if not isinstance(self.behavior, BlackBoxBehavior):
            raise TypeError("Behavior '%r' must be of type 'BlackBoxBehavior'"
                            % self.behavior)
        self.n_params = self.behavior.get_n_params()

        # Initialize optimizer
        if isinstance(self.optimizer, dict):
            try:
                # Try to generate the constructor parameter 'initial_params'
                if (not "initial_params" in self.optimizer or
                        self.optimizer["initial_params"] is None):
                    self.optimizer["initial_params"] = \
                        self.behavior.get_params()
                self.optimizer = from_dict(self.optimizer)
            except TypeError:
                # We did not specify that an optimizer must take the argument
                # 'initial_params', hence we cannot assume that it exists.
                del self.optimizer["initial_params"]
                self.optimizer = from_dict(self.optimizer)
        if not isinstance(self.optimizer, ContextualOptimizer):
            raise TypeError("Optimizer '%r' must be of type 'Optimizer'"
                            % self.optimizer)
        self._init_optimizer()

        # Set parameters
        if len(self.metaparameter_keys) != len(self.metaparameter_values):
            raise ValueError("Metaparameter keys and values must have the "
                             "same length. There are %s keys and %s "
                             "parameters." % (len(self.metaparameter_keys),
                                              len(self.metaparameter_values)))
        self.behavior.set_meta_parameters(self.metaparameter_keys,
                                          self.metaparameter_values)
        self.params = np.zeros(self.n_params)

    def get_next_behavior(self):
        if not self.is_fixed():
            self.optimizer.get_next_parameters(self.params)
            self.behavior.set_params(self.params)
            self.behavior.reset()
        return self.behavior

    def get_best_behavior(self):
        self.behavior.set_params(self.optimizer.get_best_parameters())
        self.behavior.reset()
        return self.behavior

    def set_evaluation_feedback(self, feedbacks):
        if not self.is_fixed():
            self.optimizer.set_evaluation_feedback(feedbacks)

    def is_behavior_learning_done(self):
        return self.optimizer.is_behavior_learning_done()


class BlackBoxSearch(BlackBoxSearchMixin, PickableMixin, FixableMixin,
                     BehaviorSearch):
    """Combine a black box optimizer with a black box behavior.

    Black box in this context means that only a fixed number of parameters
    will optimized with respect to a scalar reward function.

    Parameters
    ----------
    behavior : dict or Behavior subclass
        A black box behavior that is given directly or fully specified by
        a configuration dictionary.

    optimizer : dict or ContextualOptimizer subclass
        A black box optimizer that is given directly or fully specified by
        a configuration dictionary.

    metaparameter_keys : list, optional (default: [])
        Names of metaparameters for the behavior that will be set during
        initialization.

    metaparameter_values : list, optional (default: [])
        Values of metaparameters for the behavior that will be set during
        initialization.
    """

    def init(self, n_inputs, n_outputs, _=0):
        super(BlackBoxSearch, self)._init_helper(n_inputs, n_outputs)

    def _init_optimizer(self):
        if not isinstance(self.optimizer, Optimizer):
            raise TypeError(
                "BlackBoxSearch cannot be used with contextual optimizer.")
        self.optimizer.init(self.n_params)


class JustOptimizer(BlackBoxSearch):
    """Wrap only the optimizer.

    Internally, we use a behavior that only returns the parameter vector that
    has been generated by the optimizer.

    Parameters
    ----------
    optimizer : dict or Optimizer
        A black-box optimizer that is given directly or fully specified by
        a configuration dictionary.

    n_params : int, optional (default: len(optimizer.initial_params))
        Number of parameters to optimize
    """
    def __init__(self, optimizer, n_params=-1):
        kwargs = {}
        if hasattr(optimizer, "initial_params"):
            kwargs["initial_params"] = optimizer.initial_params
        elif n_params >= 0:
            kwargs["num_outputs"] = n_params
        behavior = DummyBehavior(**kwargs)
        super(JustOptimizer, self).__init__(behavior, optimizer)


class ContextualBlackBoxSearch(BlackBoxSearchMixin, PickableMixin, FixableMixin,
                               ContextualBehaviorSearch):
    """Combine a contextual black box optimizer with a black box behavior.

    Black box in this context means that only a fixed number of parameters
    will optimized with respect to a scalar reward function.

    Parameters
    ----------
    behavior : dict or Behavior subclass
        A black-box behavior that is given directly or fully specified by
        a configuration dictionary.

    optimizer : dict or ContextualOptimizer subclass
        A black-box optimizer that is given directly or fully specified by
        a configuration dictionary.

    metaparameter_keys : list, optional (default: [])
        Names of metaparameters for the behavior that will be set during
        initialization.

    metaparameter_values : list, optional (default: [])
        Values of metaparameters for the behavior that will be set during
        initialization.
    """

    def init(self, n_inputs, n_outputs, context_dims):
        self.context_dims = context_dims
        super(ContextualBlackBoxSearch, self)._init_helper(n_inputs, n_outputs)

    def _init_optimizer(self):
        if isinstance(self.optimizer, Optimizer):
            raise TypeError("ContextualBlackBoxSearch cannot be used with "
                            "non-contextual optimizer '%r'." % self.optimizer)
        self.optimizer.init(self.n_params, self.context_dims)

    def get_desired_context(self):
        """Chooses desired context for next evaluation.

        Returns
        -------
        context : ndarray-like, default=None
            The context in which the next rollout shall be performed. If None,
            the environment may select the next context without any
            preferences.
        """
        return self.optimizer.get_desired_context()

    def set_context(self, context):
        """Set context of next evaluation"""
        super(ContextualBlackBoxSearch, self).set_context(context)
        self.optimizer.set_context(context)


class JustContextualOptimizer(ContextualBlackBoxSearch):
    """Wrap only the contextual optimizer.

    Internally, we use a behavior that only returns the parameter vector that
    has been generated by the optimizer.

    Parameters
    ----------
    optimizer : dict or ContextualOptimizer
        A contextual optimizer that is given directly or fully specified by
        a configuration dictionary.

    n_params : int, optional (default: len(optimizer.initial_params))
        Number of parameters to optimize
    """
    def __init__(self, optimizer, n_params=-1):
        behavior = DummyBehavior(num_outputs=n_params)
        super(JustContextualOptimizer, self).__init__(behavior, optimizer)
        if n_params < 1:
            n_params = len(self.optimizer.initial_params)
            behavior.num_outputs = n_params
