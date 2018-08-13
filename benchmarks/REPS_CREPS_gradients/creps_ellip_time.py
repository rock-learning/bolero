"""
C-REPS

Compares the runtime performance of minimizing the dual function by using
a numerical approximation of the gradient vs computing the gradient analytically.

Results on local machine, for n_generations = 100, n_trials = 5
	-Numerical approximation: mean 9.79 seconds to complete
	-Analytical computation: mean 3.71 seconds to complete
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from creps import CREPSOptimizer
import time

def objective(x, s):
    x_offset = x + s.dot(np.array([[0.2]])).dot(s)
    return -np.array([x_offset.dot(x_offset)])

random_state = np.random.RandomState(0)
initial_params = 4.0 * np.ones(1)
n_samples_per_update = 30
variance = 0.1
context_features = "quadratic"
creps_nume = CREPSOptimizer(
    initial_params=initial_params, n_samples_per_update=n_samples_per_update,
    train_freq=n_samples_per_update, variance=variance, epsilon=2.0,
    context_features=context_features, random_state=0, approx_grad = True)
creps_analy = CREPSOptimizer(
    initial_params=initial_params, n_samples_per_update=n_samples_per_update,
    train_freq=n_samples_per_update, variance=variance, epsilon=2.0,
    context_features=context_features, random_state=0, approx_grad = False)
opts = {"Numerical gradient": creps_nume, "Analytical gradient": creps_analy}
for opt in opts.values():
    opt.init(1, 1)

n_generations = 100
n_trials = 5

params = np.empty(1)
for opt_name, opt in opts.items():
    s = time.time()
    for trial in xrange(n_trials):
        print 'Trial', trial
        for it in range(n_generations):
            contexts = random_state.rand(n_samples_per_update, 1) * 10.0 - 5.0
            for i in range(n_samples_per_update):
                opt.set_context(contexts[i])
                opt.get_next_parameters(params)
                f = objective(params, contexts[i])
                opt.set_evaluation_feedback(f)
    t = (time.time() - s) / n_trials
    print opt_name, 'average time', t, 'seconds'
