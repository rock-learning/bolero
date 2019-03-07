"""
REPS benchmark

Validates the implementation of the analytical computation of the gradient by
comparing it to the previous implementation using numerical approximation.

Compares the runtime performance of both implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from bolero.environment.objective_functions import Rosenbrock
from bolero.optimizer import REPSOptimizer
from reps_numerical import REPSOptimizerNumerical


def eval_loop(Opt, opt, n_dims, n_iter):
    x = np.empty(n_dims)
    opt.init(n_dims)
    objective = Opt(0, n_dims)
    results = np.empty(n_iter)
    for i in xrange(n_iter):
        opt.get_next_parameters(x)
        results[i] = objective.feedback(x)
        opt.set_evaluation_feedback(results[i])
    return results - objective.f_opt


n_dims = 10
n_iter = 1000
n_trials = 5

x = np.zeros(n_dims)

optimizers = {
    "Numerical gradient": REPSOptimizerNumerical(x, random_state=0),
    "Analytical gradient": REPSOptimizer(x, random_state=0),
    }
linestyles = {
    "Numerical gradient": '-',
    "Analytical gradient": '--',
}

plt.figure(figsize=(8, 6))
for name, opt in optimizers.items():
    start_time = time.time()
    for i in range(n_trials):
        r = eval_loop(Rosenbrock, opt, n_dims, n_iter)
    total_time = time.time() - start_time
    print("%s: completed in average time of %.3f seconds."
          % (name, total_time / n_trials))
    rwds = -np.maximum.accumulate(r)
    print("%s: minimum found was %f." % (name, rwds[-1]))
    plt.plot(rwds, linestyle=linestyles[name], label=name)
plt.xlabel("Function evaluations")
plt.ylabel("$f(x)$")
plt.title("Rosenbrock function")
plt.yscale("log")
plt.legend(loc='best')
plt.show()
