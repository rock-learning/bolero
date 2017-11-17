import numpy as np
from bolero.optimizer import CREPSOptimizer, CCMAESOptimizer
from nose.tools import assert_less


def target_function(x, c):
    return -(x[0] - c[0]) ** 2


def eval_loop(x, opt, n_evals=300, fixed_set=True, baseline_fct=lambda x: 0):
    results = np.ndarray(n_evals)
    if fixed_set:
        contexts = np.linspace(-1.0, 1.0, 10)[:, np.newaxis]
    for i in xrange(n_evals):
        c = opt.get_desired_context()
        if c is None:
            c = (contexts[i % contexts.shape[0]] if fixed_set else
                 np.random.uniform(-1.0, 1.0, (1,)))
        opt.set_context(c)
        opt.get_next_parameters(x)
        results[i] = target_function(x, c)
        opt.set_evaluation_feedback(results[i] + baseline_fct(c))
    return results


def test_creps():
    x = np.zeros(1)
    opt = CREPSOptimizer(x, random_state=0)
    opt.init(1, 1)
    r = eval_loop(x, opt)
    assert_less(-1e-8, r.max())
    return r


def test_creps_variance():
    x = np.zeros(1)
    opt = CREPSOptimizer(x, variance=100.0, random_state=0)
    opt.init(1, 1)
    r = eval_loop(x, opt)
    assert_less(-1e-10, r.max())
    return r


def test_creps_baseline():
    x = np.zeros(1)
    opt = CREPSOptimizer(x, context_features="quadratic", random_state=0)
    opt.init(1, 1)
    r = eval_loop(x, opt, baseline_fct=lambda x: x**2)
    assert_less(-1e-7, r.max())
    return r


def test_ccmaes():
    x = np.zeros(1)
    opt = CCMAESOptimizer(x, random_state=0)
    opt.init(1, 1)
    r = eval_loop(x, opt, n_evals=1000)
    assert_less(-1e-7, r.max())
    return r
