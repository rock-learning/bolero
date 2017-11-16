"""
========================================================
Contextual Covariance Matrix Adaption Evolution Strategy
========================================================

TODO
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from bolero.optimizer import CCMAESOptimizer, CREPSOptimizer


G = np.array([[2.0]])
def objective(x, s):
    x_offset = x + G.dot(s)
    return -np.array([x_offset.dot(x_offset)])


def plot_objective():
    contexts, params = np.meshgrid(np.arange(-6, 6, 0.1), np.arange(-6, 6, 0.1))
    rewards = np.array([[
        objective(np.array([params[i, j]]), np.array([contexts[i, j]]))[0]
        for i in range(contexts.shape[0])] for j in range(contexts.shape[1])])
    plt.contourf(params, contexts, rewards, cmap=plt.cm.Blues,  # TODO mixed up x and y?
                 levels=np.linspace(rewards.min(), rewards.max(), 30))
    plt.setp(plt.gca(), xticks=(), yticks=(), xlim=(-5, 5), ylim=(-5, 5))


def plot_policy(policy, opt_name):
    contexts = np.arange(-6, 6, 0.1)
    params = np.array([policy(np.array([s]), explore=False) for s in contexts])
    rewards = np.array([objective(p, np.array([s]))[0]
                        for p, s in zip(params, contexts)])
    print(np.mean(rewards))
    #print(rewards)
    plt.plot(contexts, params, label=opt_name + " estimate")


random_state = np.random.RandomState(0)
initial_params = 4.0 * np.ones(1)
n_samples_per_update = 20
variance = 1.0
context_features = "affine"
ccmaes = CCMAESOptimizer(
    initial_params=initial_params, n_samples_per_update=n_samples_per_update,
    variance=variance, context_features=context_features, random_state=0)
creps = CREPSOptimizer(
    initial_params=initial_params, n_samples_per_update=n_samples_per_update,
    train_freq=n_samples_per_update, variance=variance, epsilon=0.5,
    context_features=context_features, random_state=0)
opts = [ccmaes, creps]
opt_names = ["C-CMA-ES", "C-REPS"]
for opt in opts:
    opt.init(1, 1)
n_generations = 16
n_rows = 4
params = np.empty(1)
plt.figure(figsize=(n_generations * 3 / n_rows, 3 * n_rows))
for it in range(n_generations):
    plt.subplot(n_rows, n_generations / n_rows, it + 1)
    plot_objective()

    contexts = random_state.rand(n_samples_per_update, 1) * 10.0 - 5.0
    for opt_idx, opt in enumerate(opts):
        for i in range(n_samples_per_update):
            opt.set_context(contexts[i])
            opt.get_next_parameters(params)
            f = objective(params, contexts[i])
            opt.set_evaluation_feedback(f)

        policy = opt.best_policy()
        plot_policy(policy, opt_names[opt_idx])

        params = np.array(opt.history_theta).ravel()
        weights = opt.weights
        if opt_idx == 0:
            marker = "_"
        else:
            marker = "|"
            weights = opt.weights / np.sum(opt.weights)
        plt.scatter(contexts.ravel(), params, c=weights,
                    cmap=plt.cm.gray, marker=marker, s=100,
                    label=opt_names[opt_idx] + " samples")
        if it == 0:
            plt.legend(loc="lower left")
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.show()