from .environment import Environment


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
        return np.array([np.clip(int(values[0]), 0, self.n - 1)])


class HighLowHandler(object):
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, values):
        values = np.clip(values, self.matrix[:, 0], self.matrix[:, 1])
        for i in range(len(values)):
            values[i] = round(values[i], self.matrix[i, 2])


class OpenAiGym(Environment):
    def __init__(self, env_name, verbose=0):
        self.env_name = env_name
        self.verbose = verbose

    def init(self):
        try:
            import gym
        except ImportError:
            raise ImportError("OpenAiGym environment requires the Python "
                              "package 'gym'.")

        self.env = gym.make(self.env_name)
        self.n_inputs, self.input_handler = self._init_space(
            self.env.action_space)
        self.inputs = np.empty(self.n_inputs)
        self.n_outputs, _ = self._init_space(self.env.observation_space)
        self.outputs = np.empty(self.n_outputs)

    def _init_space(self, space):
        if not isinstance(space, gym.Space):
            raise ValueError("Unknown space, type '%s'" % type(space))
        elif isinstance(space, gym.spaces.Box):
            n_dims = space.shape[0]
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
        self.outputs[:] = self.env.reset()
        self.rewards = []
        self.done = False

    def get_num_inputs(self):
        return self.n_inputs

    def get_num_outputs(self):
        return self.n_outputs

    def get_outputs(self, values):
        values[:] = self.outputs

    def set_inputs(self, values):
        self.inputs[:] = self.input_handler(values)

    def step_action(self):
        self.outputs[:], reward, done, info = self.step(self.inputs)
        self.rewards.append(reward)
        self.done = self.done or done
        if self.verbose:
            print(info)

    def is_evaluation_done(self):
        return self.done

    def get_feedback(self):
        return np.asarray(self.rewards)

    def is_behavior_learning_done(self):
        False

    def get_maximum_feedback(self):
        return np.inf
