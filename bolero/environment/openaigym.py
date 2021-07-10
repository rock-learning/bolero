import numpy as np
from itertools import product
from .environment import Environment
from ..utils.log import get_logger
try:
    import gym
    gym_available = True
except ImportError:
    gym_available = False


class BoxClipHandler(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def is_discrete(self):
        return False

    def __call__(self, values):
        return np.clip(values, self.low, self.high)


class IntHandler(object):
    def __init__(self, n):
        self.n = n

    def is_discrete(self):
        return True

    def __call__(self, values):
        assert values.shape[0] == 1
        return np.clip(int(values[0]), 0, self.n - 1)


class TupleHandler(object):
    def __init__(self, handlers, n_dims):
        self.handlers = handlers
        self.n_dims = n_dims

    def is_discrete(self):
        return all([h.is_discrete() for h in self.handlers])

    def __call__(self, values):
        start = 0
        outputs = []
        for h, s in zip(self.handlers, self.n_dims):
            outputs.append(h(values[start:start + s]))
            start += s
        return outputs


class OpenAiGym(Environment):
    """Wrapper for OpenAI Gym environments.

    gym is a dependency and it is not installed with BOLeRo by default.
    You have to install it manually with "sudo pip install gym".

    OpenAI Gym is a toolkit for developing and comparing reinforcement
    learning algorithms. See `OpenAI gym's documentation
    <https://gym.openai.com/>`_ for details.

    Parameters
    ----------
    env_name : string, optional (default: 'CartPole-v0')
        Name of the environment. See `OpenAI gym's environments
        <https://gym.openai.com/envs>`_ for an overview.

    render : bool, optional (default: False)
        Visualize the environment

    log_to_file: optional, boolean or string (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: optional, boolean (default: False)
        Log to standard output

    seed : int, optional (default: None)
        Seed for the environment
    """
    def __init__(self, env_name="CartPole-v0", render=False, log_to_file=False,
                 log_to_stdout=False, seed=None):
        if not gym_available:
            raise ImportError(
                "OpenAiGym environment requires the Python package 'gym'.")
        self.env_name = env_name
        self.render = render
        self.log_to_file = log_to_file
        self.log_to_stdout = log_to_stdout
        self.seed = seed

    def init(self):
        if hasattr(gym, "configuration"):
            gym.configuration.undo_logger_setup()
        # do nothing otherwise, gym no longer modifies global logging config

        self.env = gym.make(self.env_name)
        self.n_inputs, self.input_handler = self._init_space(
            self.env.action_space)
        self.inputs = np.empty(self.n_inputs)
        self.n_outputs, _ = self._init_space(self.env.observation_space)
        self.outputs = np.empty(self.n_outputs)

        if self.seed is not None:
            self.env.seed(self.seed)

        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)

        self.step_limit = np.inf

        if self.log_to_stdout or self.log_to_file:
            self.logger.info("Number of inputs: %d" % self.n_inputs)
            self.logger.info("Number of outputs: %d" % self.n_outputs)

    def _init_space(self, space):
        if not isinstance(space, gym.Space):
            raise ValueError("Unknown space, type '%s'" % type(space))
        elif isinstance(space, gym.spaces.Box):
            n_dims = np.product(space.shape)
            handler = BoxClipHandler(space.low, space.high)
        elif isinstance(space, gym.spaces.Discrete):
            n_dims = 1
            handler = IntHandler(space.n)
        elif isinstance(space, gym.spaces.Tuple):
            n_dims = 0
            handlers = []
            n_dims_per_subspace = []
            for subspace in space.spaces:
                n_dims_subspace, subspace_handler = self._init_space(subspace)
                n_dims += n_dims_subspace
                handlers.append(subspace_handler)
                n_dims_per_subspace.append(n_dims_subspace)
            handler = TupleHandler(handlers, n_dims_per_subspace)
        else:
            raise NotImplementedError("Space of type '%s' is not supported"
                                      % type(space))
        return n_dims, handler

    def reset(self):
        if hasattr(self.env.spec, "timestep_limit"):
            self.step_limit = self.env.spec.timestep_limit
        elif hasattr(self.env.spec, "max_episode_steps"):
            self.step_limit = self.env.spec.max_episode_steps
        if self.step_limit is None:
            self.step_limit = np.inf

        self.outputs[:] = np.asarray(self.env.reset()).ravel()
        self.rewards = []
        self.done = False
        self.step = 0
        if self.render:
            self.env.render()

    def get_num_inputs(self):
        return self.n_inputs

    def get_num_outputs(self):
        return self.n_outputs

    def get_outputs(self, values):
        values[:] = self.outputs

    def set_inputs(self, values):
        self.inputs[:] = values

    def step_action(self):
        inputs = self.input_handler(self.inputs)
        observations, reward, done, info = self.env.step(inputs)
        self.outputs[:] = np.atleast_1d(observations).ravel()
        self.rewards.append(reward)
        self.done = self.done or done

        self.step += 1

        if self.log_to_stdout or self.log_to_file:
            self.logger.info(str(info))
        if self.render:
            self.env.render()

    def is_evaluation_done(self):
        return self.done or self._step_limit_reached()

    def _step_limit_reached(self):
        return self.step >= self.step_limit

    def get_feedback(self):
        feedbacks = np.asarray(self.rewards)
        self.rewards = []
        return feedbacks

    def is_behavior_learning_done(self):
        return False

    def get_maximum_feedback(self):
        if self.env.spec.reward_threshold is None:
            return np.inf
        else:
            return self.env.spec.reward_threshold

    def get_discrete_action_space(self, space=None):
        """Get list of possible actions.

        An error will be raised if the action space is not easily discretized.
        The environment must be initialized before this method can be called.

        Returns
        -------
        action_space : iterable
            Actions that the agent can take
        """
        if space is None:
            space = self.env.action_space

        if isinstance(space, gym.spaces.Discrete):
            return list(range(space.n))
        elif isinstance(space, gym.spaces.Tuple):
            subspaces = [self.get_discrete_action_space(s)
                         for s in space.spaces]
            return list(product(*subspaces))
        else:
            raise TypeError("gym environment '%s' does not have a discrete "
                            "action space" % self.env_name)
            # ... or a conversion is not yet implemented
