"""
C-REPS benchmark

Validates the implementation of the analytical computation of the gradient by
comparing it to the previous implementation using numerical approximation.

Compares the runtime performance of both implementations.
"""

import numpy as np
from creps import CREPSOptimizer
import matplotlib.pyplot as plt
import time
import pdb

def objective(x, s):
    # pdb.set_trace()
    x_offset = x + s.dot(np.array([[0.2], [0.1], [0.3]]))**2
    return -np.array([x_offset.dot(x_offset)])

nx = 1 # State size
nc = 3 # Context size

random_state = np.random.RandomState(0)
initial_params = 4.0 * np.ones(nx)
n_samples_per_update = 400
variance = 0.01
context_features = "quadratic"

creps_numerical = CREPSOptimizer( # Numerical computation
    initial_params=initial_params, n_samples_per_update=n_samples_per_update,
    train_freq=n_samples_per_update, variance=variance, epsilon=2.0,
    context_features=context_features, random_state=0, approx_grad = True)
creps_analytical = CREPSOptimizer( # Analytical computation
    initial_params=initial_params, n_samples_per_update=n_samples_per_update,
    train_freq=n_samples_per_update, variance=variance, epsilon=2.0,
    context_features=context_features, random_state=0)
opts = {"Numerical gradient": creps_numerical, "Analytical gradient": creps_analytical}

n_generations = 20
n_trials = 3 # Run time performance averaged over this number of trials

params = np.empty(nx)
rewards = dict([(k, []) for k in opts.keys()])
times = dict([(k, 0) for k in opts.keys()])
test_contexts = np.mgrid[-3:3:2, -3:3:2, -3:3:2].reshape(nc,-1).T
colors = {"Numerical gradient": "r", "Analytical gradient": "g"}

for trial in xrange(n_trials):
    # Reset opt object
    for opt in opts.values():
        opt.init(nx, nc)

    for it in range(n_generations):
        print '.'
        contexts = random_state.rand(n_samples_per_update, nc) * 10.0 - 5.0
        for opt_name, opt in opts.items():
            # Optimization...
            for i in range(n_samples_per_update):
                opt.set_context(contexts[i, :])
                opt.get_next_parameters(params)
                f = objective(params, contexts[i, :])
                start_time = time.time()
                opt.set_evaluation_feedback(f)
                times[opt_name] += time.time() - start_time

            # Only plot first trial
            if trial == 0:
                policy = opt.best_policy()
                test_params = np.array([policy(test_contexts[i,:], explore=False)
                                        for i in xrange(test_contexts.shape[0])])
                mean_reward = np.mean(
                    np.array([objective(p, s)[0]
                              for p, s in zip(test_params, test_contexts)]))
                rewards[opt_name].append(mean_reward)

                if it == n_generations - 1:
                    print opt_name, 'maximum found', rewards[opt_name][-1]

# Display computed runtime performance
for opt_name, t in times.items():
    print opt_name, 'average time', round(t / n_trials, 4), 'seconds'

# Plot results of both implementations
plt.figure()
for opt_name, values in rewards.items():
    plt.plot(values, label=opt_name, color=colors[opt_name])
plt.legend(loc="lower right")
plt.xlabel("Generation")
plt.ylabel("Mean reward")
plt.show()
