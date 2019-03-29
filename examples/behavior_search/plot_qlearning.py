"""
==========
Q-Learning
==========

A simple problem with a discrete state and action space is solved with
a tabular Q-Learning. The plot shows the obtained return
for each episode. Successful episodes terminate with the return 1, otherwise
the return is 0. The learning process is stopped when the value function
converged.
"""
print(__doc__)

import matplotlib.pyplot as plt
from bolero.environment import OpenAiGym
from bolero.behavior_search import QLearning
from bolero.controller import StepBasedController


env = OpenAiGym("FrozenLake8x8-v0", render=True, seed=1)
env.init()
bs = QLearning(env.get_discrete_action_space(), epsilon=0.2, random_state=1)
ctrl = StepBasedController(
    environment=env, behavior_search=bs, n_episodes=10000,
    finish_after_convergence=True, verbose=0)
rewards = ctrl.learn()

plt.figure()
ax = plt.subplot(111)
ax.set_title("Learning progress")
ax.plot(rewards)
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
plt.show()
