import numpy as np
from bolero.optimizer.creps import solve_dual_contextual_reps, CREPSOptimizer
from bolero.representation.context_transformations import quadratic
from nose.tools import assert_raises_regexp, assert_true, assert_equal


def test_shapes_mismatch():
    S = np.linspace(0, 1, 11)[:, np.newaxis]
    R = np.zeros(10)
    assert_raises_regexp(ValueError, "Number of contexts .* number of returns",
                         solve_dual_contextual_reps, S, R, 1.0, 0.0)


def test_baseline_overestimates():
    x = np.linspace(0, 1, 11)
    S = np.array([quadratic(s) for s in x[:, np.newaxis]])
    R = 10 * (x - 0.3) ** 2
    min_eta = 1e-6
    d, eta, nu = solve_dual_contextual_reps(S, R, 1.0, min_eta)
    R_baseline = S.dot(nu)
    assert_equal(eta, min_eta)
    assert_true(np.all(R_baseline >= R))
    assert_true(np.all(d >= 0))
    assert_equal(np.sum(d), 1.0)


def test_cmaes_dimensions_mismatch():
    opt = CREPSOptimizer(initial_params=np.zeros(5))
    assert_raises_regexp(ValueError, "Number of dimensions", opt.init, 10, 2)
