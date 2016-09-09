"""
=========
Cart Pole
=========

This is an example of how to use the Cart Pole environment from OpenAI Gym
via the wrapper that is provided with BOLeRo. A linear policy is sufficient
to solve the problem and policy search algorithm usually work very well in
this domain.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from bolero.environment import OpenAiGym
from bolero.behavior_search import BlackBoxSearch
from bolero.optimizer import CMAESOptimizer
from bolero.representation import LinearBehavior
from bolero.controller import Controller


beh = LinearBehavior()
env = OpenAiGym("CartPole-v0", max_steps=200, render=False, seed=0)
opt = CMAESOptimizer(variance=100.0 ** 2, random_state=0)
bs = BlackBoxSearch(beh, opt)
controller = Controller(environment=env, behavior_search=bs, n_episodes=500)

rewards = controller.learn()
controller.episode_with(bs.get_best_behavior())

plt.figure()
ax = plt.subplot(111)
ax.set_title("Optimization progress")
ax.plot(rewards)
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_ylim(-10, 210)
plt.show()
