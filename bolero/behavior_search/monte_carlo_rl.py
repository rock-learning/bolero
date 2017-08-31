import numpy as np
from collections import defaultdict
from .behavior_search import BehaviorSearch, PickableMixin
from ..representation import Behavior
from ..utils import check_random_state


class EpsilonGreedyPolicy(Behavior):
    """Epsilon-greedy policy.

    A random action is selected with probability 'epsilon'. Otherwise
    the optimum action based on the current estimate of Q is selected.

    Parameters
    ----------
    Q : dict
        State-action value function

    action_space : list
        Actions that the agent can select from

    epsilon : float, optional (default: 0.1)
        Probability of selecting a random action

    random_state : int or RandomState, optional (default: None)
        Seed for the random number generator or RandomState object.
    """
    def __init__(self, Q, action_space, epsilon=0.1, random_state=None):
        self.Q = Q
        self.action_space = action_space
        self.epsilon = epsilon
        self.random_state = check_random_state(random_state)

    def init(self, n_inputs, n_outputs):
        """Initialize the behavior.

        Parameters
        ----------
        n_inputs : int
            number of inputs

        n_outputs : int
            number of outputs
        """
        assert n_inputs == 1, "discrete state space required"
        assert n_outputs == 1, "discrete action space required"
        self.visited_states = []
        self.actions_taken = []

    def set_meta_parameters(self, keys, meta_parameters):
        """Set meta-parameters.

        Meta-parameters could be the goal, obstacles, ...

        Parameters
        ----------
        keys : list of string
            names of meta-parameters
        meta_parameters : list of lists of float values
            One list of floats for each parameter          
        """

    def set_inputs(self, inputs):
        """Set input for the next step.

        If the input vector consists of positions and derivatives of these,
        by convention all positions and all derivatives should be stored
        contiguously.

        Parameters
        ----------
        inputs : array-like, shape = (n_inputs,)
            inputs, e.g. current state of the system
        """
        self.s = int(inputs[0])
        self.visited_states.append(self.s)

    def get_outputs(self, outputs):
        """Get outputs of the last step.

        If the output vector consists of positions and derivatives of these,
        by convention all positions and all derivatives should be stored
        contiguously.

        Parameters
        ----------
        outputs : array-like, shape = (n_outputs,)
            outputs, e.g. next action, will be updated
        """
        outputs[0] = self.action
        self.actions_taken.append(self.action)

    def step(self):
        """Compute output for the received input.

        Uses the inputs and meta-parameters to compute the outputs.
        """
        if self.random_state.rand() < self.epsilon:
            self.action = self.random_state.choice(self.action_space)
        else:
            self._select_best_action()

    def _select_best_action(self):
        Qs = np.array([self.Q[self.s][a] for a in self.action_space])
        best_actions = np.where(Qs == max(Qs))[0]
        self.action = self.action_space[
            self.random_state.choice(best_actions)]

    def can_step(self):
        """Returns if step() can be called again.

        Returns
        -------
        can_step : bool
            Can we call step() again?
        """
        return True


class MonteCarloRL(BehaviorSearch, PickableMixin):
    """Tabular Monte Carlo is a model-free reinforcement learning method.

    This implements the epsilon-soft on-policy Monte Carlo control algorithm
    shown at page 120 of "Reinforcement Learning: An Introduction"
    (Sutton and Barto, 2nd edition,
    http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf).
    The action space and the state space must be discrete for this
    implementation.

    Parameters
    ----------
    action_space : list
        Actions that the agent can select from

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
    def __init__(self, action_space, gamma=0.9, epsilon=0.1, random_state=None):
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
        n_steps = len(visited_states)
        assert n_steps == len(feedbacks)
        assert n_steps == len(actions_taken)
        gammas = np.hstack(
            ((1,), np.cumprod(np.ones(n_steps - 1) * self.gamma)))
        diff = 0.0
        for t in range(n_steps):
            s = visited_states[t]
            a = actions_taken[t]
            ret = sum(feedbacks[t:] * gammas[:n_steps - t])

            self.returns[s][a].append(ret)
            last_Q = self.Q[s][a]
            self.Q[s][a] = np.mean(self.returns[s][a])
            diff = max(diff, np.abs(last_Q - self.Q[s][a]))
        self.done = any(feedbacks > 0) and diff < 1e-3

    def is_behavior_learning_done(self):
        """Check if the value function converged.

        Returns
        -------
        finished : bool
            Is the learning of a behavior finished?
        """
        return self.done

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
