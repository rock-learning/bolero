import numpy as np
import matplotlib.pyplot as plt
from bolero.behavior_search import BehaviorSearch
from bolero.representation import Behavior


class EpsilonGreedyPolicy(Behavior):  # TODO document
    def __init__(self, Q, action_space, epsilon):
        self.Q = Q
        self.action_space = action_space
        self.epsilon = epsilon

    def init(self, n_inputs, n_outputs):
        assert n_inputs == 1, "discrete state space"
        assert n_outputs == 1, "discrete action space"
        self.visited_states = []
        self.actions_taken = []

    def set_meta_parameters(self, keys, meta_parameters):
        pass

    def set_inputs(self, inputs):
        self.inputs = np.copy(inputs)
        self.visited_states.append(int(self.inputs[0]))

    def get_outputs(self, outputs):
        outputs[0] = self.action
        self.actions_taken.append(self.action)

    def step(self):
        Qs = [(a, self.Q[(s, a)]) for s, a in self.Q.keys() if s == self.inputs]
        if np.random.rand() < self.epsilon or len(Qs) == 0:
            self.action = self.action_space[np.random.randint(0, len(self.action_space))]
        else:
            max_idx = np.argmax([q for a, q in Qs])
            self.action = Qs[max_idx][0]

    def can_step(self):
        return True


class MonteCarlo(BehaviorSearch):
    """Tabular Monte Carlo is a model-free reinforcement learning.

    TODO document action and state space requirements
    TODO document parameters
    """
    def __init__(self, action_space, gamma=0.9, epsilon=0.1):
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon

    def init(self, n_inputs, n_outputs):
        self.Q = dict()
        self.policy = EpsilonGreedyPolicy(self.Q, self.action_space, self.epsilon)
        self.Return = dict()

    def get_next_behavior(self):
        self.policy.init(1, 1)  # TODO
        return self.policy

    def set_evaluation_feedback(self, feedbacks):
        visited_states = self.policy.visited_states
        actions_taken = self.policy.actions_taken
        n_steps = len(visited_states)
        assert n_steps == len(feedbacks)
        assert n_steps == len(actions_taken)
        gammas = np.hstack(((1,), np.cumprod(np.ones(n_steps - 1) * self.gamma)))
        for t in range(n_steps):
            s = visited_states[t]
            a = actions_taken[t]
            r = sum(feedbacks[t:] * gammas[:n_steps - t])
            #print("s=%d, a=%d, r=%d" % (s, a, r))
            if (s, a) not in self.Return:
                self.Return[(s, a)] = []
            self.Return[(s, a)].append(r)
            if (s, a) not in self.Q:
                self.Q[(s, a)] = None
            self.Q[(s, a)] = sum(self.Return[(s, a)]) / float(len(self.Return[(s, a)]))

    def write_results(self, result_path):
        raise NotImplementedError()

    def get_behavior_from_results(self, result_path):
        raise NotImplementedError()

    def is_behavior_learning_done(self):
        return False  # TODO check convergence

    def get_best_behavior(self):
        policy = EpsilonGreedyPolicy(self.Q, self.action_space, epsilon=0.0)
        policy.init(1, 1)  # TODO
        return policy


if __name__ == "__main__":
    from bolero.environment import OpenAiGym
    from bolero.controller import Controller
    env = OpenAiGym("FrozenLake-v0", render=True, seed=0)
    env.init()
    action_space = list(range(env.env.action_space.n))
    bs = MonteCarlo(action_space)
    ctrl = Controller(environment=env, behavior_search=bs, n_episodes=2000,
                      verbose=0)
    rewards = ctrl.learn()

    print(ctrl.episode_with(bs.get_best_behavior()))

    for s in range(14):
        for a in range(4):
            if (s, a) in bs.Q:
                print("(%d, %d) -> %.3f" % (s, a, bs.Q[s, a]))

    plt.figure()
    ax = plt.subplot(111)
    ax.set_title("Optimization progress")
    ax.plot(rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    plt.show()
