"""
C-REPS

Validates the implementation of the analytical computation of the gradient by
comparing it to the previous implementation using numerical approximation.

Compares the runtime performance of both implementations.
"""

import numpy as np
from creps import CREPSOptimizer
import matplotlib.pyplot as plt
import time

def objective(x, s):
    x_offset = x + s.dot(np.array([[0.2]])).dot(s)
    return -np.array([x_offset.dot(x_offset)])

random_state = np.random.RandomState(0)
initial_params = 4.0 * np.ones(1)
n_samples_per_update = 30
variance = 0.03
context_features = "quadratic"

creps_numerical = CREPSOptimizer( # Numerical computation
    initial_params=initial_params, n_samples_per_update=n_samples_per_update,
    train_freq=n_samples_per_update, variance=variance, epsilon=2.0,
    context_features=context_features, random_state=0, approx_grad = True)
creps_analytical = CREPSOptimizer( # Analytical computation
    initial_params=initial_params, n_samples_per_update=n_samples_per_update,
    train_freq=n_samples_per_update, variance=variance, epsilon=2.0,
    context_features=context_features, random_state=0, approx_grad = False)
opts = {"Numerical gradient": creps_numerical, "Analytical gradient": creps_analytical}

n_generations = 16
n_trials = 5 # Run time performance averaged over this number of trials

params = np.empty(1)
rewards = dict([(k, []) for k in opts.keys()])
times = dict([(k, 0) for k in opts.keys()])
test_contexts = np.arange(-6, 6, 0.1)
colors = {"Numerical gradient": "r", "Analytical gradient": "g"}

for trial in xrange(n_trials):
    # Reset opt object
    for opt in opts.values():
        opt.init(1, 1)

    for it in range(n_generations):
        contexts = random_state.rand(n_samples_per_update, 1) * 10.0 - 5.0
        for opt_name, opt in opts.items():
            start_time = time.time()

            # Optimization...
            for i in range(n_samples_per_update):
                opt.set_context(contexts[i])
                opt.get_next_parameters(params)
                f = objective(params, contexts[i])
                opt.set_evaluation_feedback(f)

            # Log time
            times[opt_name] += time.time() - start_time

            # Only plot first trial
            if trial == 0:
                policy = opt.best_policy()
                test_params = np.array([policy(np.array([s]), explore=False)
                                        for s in test_contexts])
                mean_reward = np.mean(
                    np.array([objective(p, np.array([s]))[0]
                              for p, s in zip(test_params, test_contexts)]))
                rewards[opt_name].append(mean_reward)

# Display computed runtime performance
for opt_name, t in times.items():
    print opt_name, 'average time', round(t / n_trials, 2), 'seconds'

# Plot results of both implementations
plt.figure()
for opt_name, values in rewards.items():
    plt.plot(values, label=opt_name, color=colors[opt_name])
plt.legend(loc="lower right")
plt.xlabel("Generation")
plt.ylabel("Mean reward")
plt.show()
