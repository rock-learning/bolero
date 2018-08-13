"""
C-REPS

Compares the runtime performance of minimizing the dual function by using
a numerical approximation of the gradient vs computing the gradient analytically.

Results on local machine, for n_dims = 300, n_iter = 100, n_trials = 20
	-Numerical approximation: mean 4.49 seconds to complete
	-Analytical computation: mean 4.59 seconds to complete

However as demonstrated in reps_rosen.png the analyticial gradient led to a better solution.
"""

import numpy as np
from reps import REPSOptimizer
from bolero.environment.objective_functions import Rosenbrock
import matplotlib.pyplot as plt
import time

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

n_dims = 300
n_iter = 100
n_trials = 5

x = np.zeros(n_dims)

optimizers = {
    "Numerical gradient": REPSOptimizer(x, random_state=0, approx_grad = True),
    "Analytical gradient": REPSOptimizer(x, random_state=0, approx_grad = False),
    }


plt.figure(figsize=(12, 8))
plt.xlabel("Function evaluations")
plt.ylabel("$f(x)$")
plt.title("Rosenbrock function")
for name, opt in optimizers.items():
    total_time = 0
    for i in range(n_trials):
        print 'Trial', i
        s = time.time()
        r = eval_loop(Rosenbrock, opt, n_dims, n_iter)
        total_time += time.time() - s
    print name, ' completed in average time of', total_time / n_trials, 'seconds'
    plt.plot(-np.maximum.accumulate(r), label=name)
plt.yscale("log")
plt.legend(loc='best')

plt.show()
