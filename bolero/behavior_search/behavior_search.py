"""BehaviorSearch interface."""

from abc import ABCMeta, abstractmethod
from tools import NonContextualException


class ContextualBehaviorSearch(object):
    """Common interface for (contextual) behavior search."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def init(self, n_inputs, n_outputs, n_context_dims):
        """Initialize the behavior search.

        Parameters
        ----------
        n_inputs : int
            number of inputs of the behavior

        n_outputs : int
            number of outputs of the behavior

        n_context_dims : int
            number of context dimensions
        """

    @abstractmethod
    def get_next_behavior(self):
        """Obtain next behavior for evaluation.

        Returns
        -------
        behavior : Behavior
            mapping from input to output
        """

    @abstractmethod
    def set_evaluation_feedback(self, feedbacks):
        """Set feedback for the last behavior.

        Parameters
        ----------
        feedbacks : list of float
            feedback for each step or for the episode, depends on the problem
        """

    @abstractmethod
    def write_results(self, result_path):
        """Store current search state.

        Parameters
        ----------
        result_path : string
            path in which the state should be stored
        """

    @abstractmethod
    def get_behavior_from_results(self, result_path):
        """Recover search state from file.

        Parameters
        ----------
        result_path : string
            path in which we search for the file
        """

    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished, e.g. it converged.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return False

    def get_desired_context(self):
        """Chooses desired context for next evaluation.

        Returns
        -------
        context : ndarray-like, default=None
            The context in which the next rollout shall be performed. If None,
            the environment may select the next context without any
            preferences.
        """
        return None

    def set_context(self, context):
        """ Set context of next evaluation.

        Note that the set context need not necessarily be the same that was
        requested by get_desired_context().

        Parameters
        ----------
        context : ndarray-like
            The context in which the next rollout will be performed
        """
        self.context = context

    @abstractmethod
    def get_best_behavior_template(self):
        """ Return current best estimate of contextual policy. """


class BehaviorSearch(ContextualBehaviorSearch):
    """BehaviorSearch interface, i.e. common interface of learning algorithm.
    """

    def init(self, n_inputs, n_outputs, n_context_dims=0):
        """Initialize the behavior search.

        Parameters
        ----------
        n_inputs : int
            number of inputs of the behavior

        n_outputs : int
            number of outputs of the behavior

        n_context_dims : int
            number of context dimensions. Restricted to 0 for non-contextual
            behavior search.
        """
        if n_context_dims > 0:
            raise ValueError("BehaviorSearch does not support contextual "
                             "problems.")
        super(BehaviorSearch, self).init(n_inputs, n_outputs, n_context_dims)

    def get_desired_context(self):
        """ Method not supported by BehaviorSearch. """
        raise NonContextualException("get_desired_context() not supported.")

    def set_context(self, context):
        """ Method not supported by BehaviorSearch. """
        raise NonContextualException("set_context() not supported.")

    @abstractmethod
    def get_best_behavior(self):
        """Returns the best behavior found so far.

        Returns
        -------
        behavior : Behavior
            mapping from input to output
        """

    def get_best_behavior_template(self):
        """ Return current best estimate of contextual policy. """
        raise NonContextualException("get_best_behavior_template() not "
                                     "supported.")
