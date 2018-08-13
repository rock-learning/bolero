"""
REPS

Validates the implementation of the analytical gradient dual function minimization by
comparing it to the existing solution using numerical approximation.
"""

import numpy as np
import matplotlib.pyplot as plt
from bolero.environment import OpenAiGym
from bolero.behavior_search import BlackBoxSearch
from reps import REPSOptimizer
from bolero.representation import LinearBehavior
from bolero.controller import Controller


beh = LinearBehavior()

env = OpenAiGym("CartPole-v0", render=False, seed=0)
opt_approx = REPSOptimizer(variance=10.0 ** 2, random_state=0)
bs_approx = BlackBoxSearch(beh, opt_approx)
controller_approx = Controller(environment=env, behavior_search=bs_approx, n_episodes=300, approx_grad = True)
rewards_approx = controller_approx.learn()

env = OpenAiGym("CartPole-v0", render=False, seed=0)
opt_analy = REPSOptimizer(variance=10.0 ** 2, random_state=0)
bs_analy = BlackBoxSearch(beh, opt_analy)
controller_analy = Controller(environment=env, behavior_search=bs_analy, n_episodes=300, approx_grad = False)
rewards_analy = controller_analy.learn()

plt.figure()
ax = plt.subplot(111)
ax.set_title("Optimization progress")
ax.plot(rewards_approx, 'b', label = "Numerical gradient")
ax.plot(rewards_analy, 'g', label = "Analytical gradient")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_ylim(-10, 210)
ax.legend()
plt.show()
