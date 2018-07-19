"""
=========================
Obstacle Avoidance ProMPs
=========================

We use CMA-ES to optimize a ProMP so that it avoids point obstacles.
"""

import numpy as np
import matplotlib.pyplot as plt
from bolero.environment import OptimumTrajectory
from bolero.behavior_search import BlackBoxSearch
from bolero.optimizer import CMAESOptimizer
from bolero.representation import ProMPBehavior
from bolero.controller import Controller

print(__doc__)

n_task_dims = 2
obstacles = [np.array([0.5, 0.5]), np.array([0.6, 0.8]), np.array([0.8, 0.6])]
x0 = np.zeros(n_task_dims)
g = np.ones(n_task_dims)
execution_time = 1.0
dt = 0.01
n_features = 5
n_episodes = 500
use_covar = True

beh = ProMPBehavior(
    execution_time, dt, n_features, learn_covariance=True, use_covar=True)

# init linear to have a guess
beh.init(4, 4)
beh.set_meta_parameters(["g", "x0"], [g, x0])
beh.imitate(np.tile(np.linspace(0, 1, 101), 2).reshape((2, 101, -1)))

env = OptimumTrajectory(
    x0,
    g,
    execution_time,
    dt,
    obstacles,
    penalty_goal_dist=10000.0,
    penalty_start_dist=10000.0,
    penalty_obstacle=1000.0,
    penalty_length=10.,
    calc_acc=True,
    use_covar=True)
opt = CMAESOptimizer(
    variance=0.1**2, random_state=0, initial_params=beh.get_params())
bs = BlackBoxSearch(beh, opt)
controller = Controller(
    environment=env,
    behavior_search=bs,
    n_episodes=n_episodes,
    record_inputs=True)

rewards = controller.learn(["x0", "g"], [x0, g])
controller.episode_with(bs.get_best_behavior(), ["x0", "g"], [x0, g])
X = np.asarray(controller.inputs_[-1])
X_hist = np.asarray(controller.inputs_)

plt.figure(figsize=(8, 5))
ax = plt.subplot(121)
ax.set_title("Optimization progress")
ax.plot(rewards)
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")

ax = plt.subplot(122, aspect="equal")
ax.set_title("Learned trajectory")
ProMPBehavior.plotCovariance(ax, X[:, :2],
                             np.array(X[:, 4:]).reshape(-1, 4, 4))

env.plot(ax)
ax.plot(X[:, 0], X[:, 1], lw=5, label="Final trajectory")
for it, X in enumerate(X_hist[::int(n_episodes / 10)]):
    ax.plot(X[:, 0], X[:, 1], c="k", alpha=it / 20.0, lw=3, ls="--")
ax.set_xticks(())
ax.set_yticks(())
plt.legend(loc="best")
plt.show()
