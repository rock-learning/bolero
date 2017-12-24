import logging
import numpy as np
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

    def __call__(self, values):
        return np.clip(values, self.low, self.high)


class IntHandler(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, values):
        assert values.shape[0] == 1
        return np.clip(int(values[0]), 0, self.n - 1)


class HighLowHandler(object):
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, values):
        values = np.clip(values, self.matrix[:, 0], self.matrix[:, 1])
        for i in range(len(values)):
            values[i] = round(values[i], self.matrix[i, 2])
        return values


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
        gym.configuration.undo_logger_setup()

        self.env = gym.make(self.env_name)
        self.n_inputs, self.input_handler = self._init_space(
            self.env.action_space)
        self.inputs = np.empty(self.n_inputs)
        self.n_outputs, _ = self._init_space(self.env.observation_space)
        self.outputs = np.empty(self.n_outputs)

        if self.seed is not None:
            self.env.seed(self.seed)

        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)

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
        elif isinstance(space, gym.spaces.HighLow):
            n_dims = space.num_rows
            handler = HighLowHandler(space.matrix)
        elif isinstance(space, gym.spaces.Tuple):
            raise NotImplementedError("Space of type '%s' is not supported"
                                      % type(space))
        return n_dims, handler

    def reset(self):
        self.outputs[:] = self.env.reset().ravel()
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
        if self.step >= self.env.spec.timestep_limit:
            self.done = True

        if self.log_to_stdout or self.log_to_file:
            self.logger.info(str(info))
        if self.render:
            self.env.render()

    def is_evaluation_done(self):
        return self.done

    def get_feedback(self):
        return np.asarray(self.rewards)

    def is_behavior_learning_done(self):
        return False

    def get_maximum_feedback(self):
        if self.env.spec.reward_threshold is None:
            return np.inf
        else:
            return self.env.spec.reward_threshold

    def get_discrete_action_space(self):
        """Get list of possible actions.

        An error will be raised if the action space of the problem is not
        discrete. The environment must be initialized before this method can
        be called.

        Returns
        -------
        action_space : iterable
            Actions that the agent can take
        """
        if not hasattr(self.env.action_space, "n"):
            raise TypeError("gym environment '%d' does not have a discrete "
                            "action space")
        return list(range(self.env.action_space.n))
