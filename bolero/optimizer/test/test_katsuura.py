"""Test for multimodal function optimization."""

import numpy as np
from nose.tools import assert_less
from bolero.optimizer import IPOPCMAESOptimizer, BIPOPCMAESOptimizer
from bolero.environment.objective_functions import Katsuura


def eval_loop(x, opt, n_dims, n_evals):
    objective = Katsuura(0, n_dims)
    results = np.empty(n_evals)
    for i in xrange(n_evals):
        opt.get_next_parameters(x)
        results[i] = objective.feedback(x)
        opt.set_evaluation_feedback(results[i])
    return results - objective.f_opt


def test_ipopcmaes(n_dims=2, n_evals=1500, **kwargs):
    x = np.zeros(n_dims)
    opt = IPOPCMAESOptimizer(x, bounds=np.array([[-5, 5]]), random_state=0,
                             **kwargs)
    opt.init(n_dims)
    r = eval_loop(x, opt, n_dims, n_evals)
    assert_less(-1e1, r.max())
    return r


def test_bipopcmaes(n_dims=2, n_evals=1500, **kwargs):
    x = np.zeros(n_dims)
    opt = BIPOPCMAESOptimizer(x, bounds=np.array([[-5, 5]]), random_state=0,
                              **kwargs)
    opt.init(n_dims)
    r = eval_loop(x, opt, n_dims, n_evals)
    assert_less(-1e1, r.max())
    return r
