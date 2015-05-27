"""Test for unimodal function optimization."""
import numpy as np
from nose.tools import assert_less
from bolero.optimizer import CMAESOptimizer
from bolero.environment.objective_functions import Rosenbrock


def eval_loop(x, opt, n_dims, n_evals=1000):
    objective = Rosenbrock(0, n_dims)
    results = np.empty(n_evals)
    for i in xrange(n_evals):
        opt.get_next_parameters(x)
        results[i] = objective.feedback(x)
        opt.set_evaluation_feedback(results[i])
    return results - objective.f_opt


def test_cmaes(n_dims=2):
    x = np.zeros(n_dims)
    opt = CMAESOptimizer(x, random_state=0, log_to_stdout=False)
    opt.init(n_dims)
    r = eval_loop(x, opt, n_dims)
    assert_less(-1e-5, r.max())
    return r


def test_acmaes(n_dims=2):
    x = np.zeros(n_dims)
    opt = CMAESOptimizer(x, active=True, random_state=0, log_to_stdout=False)
    opt.init(n_dims)
    r = eval_loop(x, opt, n_dims)
    assert_less(-1e-5, r.max())
    return r
