"""Test for multimodal function optimization."""

import numpy as np
from nose.tools import assert_greater
from bolero.optimizer import (RestartCMAESOptimizer, IPOPCMAESOptimizer,
                              BIPOPCMAESOptimizer)
from bolero.environment.objective_functions import Katsuura


def eval_loop(x, opt, n_dims, n_evals):
    objective = Katsuura(0, n_dims)
    results = np.empty(n_evals)
    for i in xrange(n_evals):
        opt.get_next_parameters(x)
        results[i] = objective.feedback(x)
        opt.set_evaluation_feedback(results[i])
    return results - objective.f_opt


def test_restartcmaes(n_dims=2, n_evals=2000):
    x = np.zeros(n_dims)
    opt = RestartCMAESOptimizer(x, bounds=np.array([[-5, 5]]), random_state=0)
    opt.init(n_dims)
    r = eval_loop(x, opt, n_dims, n_evals)
    assert_greater(r.max(), -1e5)


def test_ipopcmaes(n_dims=2, n_evals=3500):
    x = np.zeros(n_dims)
    opt = IPOPCMAESOptimizer(x, bounds=np.array([[-5, 5]]), random_state=0)
    opt.init(n_dims)
    r = eval_loop(x, opt, n_dims, n_evals)
    assert_greater(r.max(), -1e5)


def test_bipopcmaes(n_dims=1, n_evals=1000):
    x = np.zeros(n_dims)
    opt = BIPOPCMAESOptimizer(x, bounds=np.array([[-5, 5]]), random_state=0)
    opt.init(n_dims)
    r = eval_loop(x, opt, n_dims, n_evals)
    assert_greater(r.max(), -1e5)
