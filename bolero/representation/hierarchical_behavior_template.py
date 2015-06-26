# Author: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

from .behavior import BehaviorTemplate, BlackBoxBehavior
from .ul_policies import UpperLevelPolicy


class HierarchicalBehaviorTemplate(BehaviorTemplate):
    """Behavior template that consists of an upper-level policy and a behavior.

    Parameters
    ----------
    upper_level_policy : UpperLevelPolicy or callable
        Upper-level policy that selects the parameters of a black-box behavior
        for a given context

    behavior : BlackBoxBehavior
        A black-box behavior that is completely defined by a parameter vector
        of fixed size

    explore : bool, optional (default: False)
        Allow upper-level policy to be stochastic
    """
    def __init__(self, upper_level_policy, behavior, explore=False):
        if not isinstance(behavior, BlackBoxBehavior):
            raise ValueError("%r (type: %r) must be of type 'BlackBoxBehavior'"
                             % (behavior, type(behavior)))
        self.upper_level_policy = upper_level_policy
        self.behavior = behavior
        self.explore = explore

    def get_behavior(self, context):
        """Get behavior for a given context.

        Parameters
        ----------
        context : array-like, shape (n_context_dims,)
            Current context

        Returns
        -------
        behavior : BlackBoxBehavior
            Behavior for the given context
        """
        params = self.upper_level_policy(context, self.explore)
        self.behavior.set_params(params)
        self.behavior.reset()
        return self.behavior
