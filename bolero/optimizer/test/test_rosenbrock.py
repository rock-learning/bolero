"""Test for unimodal function optimization."""
import numpy as np
from nose.tools import assert_less, assert_greater
from bolero.optimizer import CMAESOptimizer, NoOptimizer, RandomOptimizer
from bolero.environment.objective_functions import Rosenbrock


n_dims = 2


def eval_loop(x, opt, n_dims, n_evals=1000):
    objective = Rosenbrock(0, n_dims)
    results = np.empty(n_evals)
    for i in xrange(n_evals):
        opt.get_next_parameters(x)
        results[i] = objective.feedback(x)
        opt.set_evaluation_feedback(results[i])
    return results - objective.f_opt


def test_cmaes():
    x = np.zeros(n_dims)
    opt = CMAESOptimizer(x, random_state=0, log_to_stdout=False)
    opt.init(n_dims)
    r = eval_loop(x, opt, n_dims)
    assert_greater(r.max(), -1e-5)


def test_acmaes():
    x = np.zeros(n_dims)
    opt = CMAESOptimizer(x, active=True, random_state=0, log_to_stdout=False)
    opt.init(n_dims)
    r = eval_loop(x, opt, n_dims)
    assert_greater(r.max(), -1e-5)


def test_no():
    x = np.zeros(n_dims)
    opt = NoOptimizer(x)
    opt.init(n_dims)
    r = eval_loop(x, opt, n_dims)
    assert_less(-200, r.max())
    assert_greater(-100, r.max())


def test_random():
    x = np.zeros(n_dims)
    opt = RandomOptimizer(x, random_state=0)
    opt.init(n_dims)
    r = eval_loop(x, opt, n_dims)
    assert_greater(r.max(), -0.1)
