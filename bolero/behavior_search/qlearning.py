import numpy as np
from collections import defaultdict
from .behavior_search import BehaviorSearch, PickableMixin
from ..representation import Behavior
from .monte_carlo_rl import EpsilonGreedyPolicy


class QLearning(BehaviorSearch, PickableMixin):
    """Q-Learning is a model-free reinforcement learning method.

    This implements the epsilon-soft off-policy TD control algorithm
    Q-learning shown at page 131 of "Reinforcement Learning: An Introduction"
    (Sutton and Barto, 2nd edition,
    http://incompleteideas.net/book/bookdraft2018mar21.pdf).
    The action space and the state space must be discrete for this
    implementation.

    Parameters
    ----------
    action_space : list
        Actions that the agent can select from

    alpha : float, optional (default: 0.1)
        The learning rate. Must be within (0, 1].

    gamma : float, optional (default: 0.9)
        Discount factor for the discounted infinite horizon model

    epsilon : float, optional (default: 0.1)
        Exploration probability for epsilon-greedy policy

    convergence_threshold : float, optional (default: 1e-3)
        Learning will be stopped if the maximum difference of the value
        function between iterations is below this threshold

    random_state : int or RandomState, optional (default: None)
        Seed for the random number generator or RandomState object.
    """
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=0.1,
                 random_state=None):
        self.alpha = alpha
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.random_state = random_state

    def init(self, n_inputs, n_outputs):
        """Initialize the behavior search.

        Parameters
        ----------
        n_inputs : int
            number of inputs of the behavior

        n_outputs : int
            number of outputs of the behavior
        """
        assert n_inputs == 1, "discrete state space required"
        assert n_outputs == 1, "discrete action space required"
        self.Q = defaultdict(lambda: dict((a, 0.0) for a in self.action_space))
        self.policy = EpsilonGreedyPolicy(
            self.Q, self.action_space, self.epsilon, self.random_state)
        self.returns = defaultdict(lambda: defaultdict(lambda: []))
        self.done = False

    def get_next_behavior(self):
        """Obtain next behavior for evaluation.

        Returns
        -------
        behavior : Behavior
            mapping from input to output
        """
        self.policy.init(1, 1)
        return self.policy

    def set_evaluation_feedback(self, feedbacks):
        """Set feedback for the last behavior.

        Parameters
        ----------
        feedbacks : list of float
            feedback for each step or for the episode, depends on the problem
        """
        visited_states = self.policy.visited_states
        actions_taken = self.policy.actions_taken

        if len(visited_states) < 2:
            return
        if len(feedbacks) < 1:
            return

        self.policy.visited_states = visited_states[1:]
        self.policy.actions_taken = actions_taken[1:]

        s = visited_states[0]
        s2 = visited_states[1]
        a = actions_taken[0]
        r = feedbacks[0]

        last_Q = self.Q[s][a]
        best_next_Q = np.max([self.Q[s2][a] for a in self.action_space])
        td_error = r + self.gamma * best_next_Q - last_Q
        self.Q[s][a] = last_Q + self.alpha * td_error

    def is_behavior_learning_done(self):
        """Check if the value function converged.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return False  # TODO find a more intelligent way to terminate...

    def get_best_behavior(self):
        """Returns the best behavior found so far.

        Returns
        -------
        behavior : Behavior
            mapping from input to output
        """
        policy = EpsilonGreedyPolicy(self.Q, self.action_space, epsilon=0.0)
        policy.init(1, 1)
        return policy