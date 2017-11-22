import numpy as np
from bolero.optimizer import SkOptOptimizer
from bolero.optimizer.skoptimize import skopt_available
if not skopt_available:
    from nose import SkipTest
    raise SkipTest("scikit-optimize is not installed")
from nose.tools import (assert_greater, assert_raises_regexp, assert_equal,
                        assert_false)


def test_bayes_opt_base_estimator():
    from skopt.learning import GaussianProcessRegressor
    from skopt.learning.gaussian_process.kernels import ConstantKernel
    from skopt.learning.gaussian_process.kernels import Matern
    cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
    matern = Matern(
        length_scale=np.ones(2), length_scale_bounds=[(0.01, 100)] * 2, nu=2.5)
    base_estimator = GaussianProcessRegressor(
        kernel=cov_amplitude * matern,
        normalize_y=True, random_state=0, alpha=0.0,
        noise="gaussian", n_restarts_optimizer=2)
    opt = SkOptOptimizer(
        dimensions=[(-1.0, 1.0), (-1.0, 1.0)], base_estimator=base_estimator,
        random_state=0)
    opt.init(2)
    params = np.empty(2)
    for _ in range(10):
        opt.get_next_parameters(params)
        feedback = [-np.linalg.norm(params - 0.5384 * np.ones(2))]
        opt.set_evaluation_feedback(feedback)
    assert_greater(opt.get_best_fitness(), -0.3)


def test_bayes_opt_gpr():
    opt = SkOptOptimizer(
        dimensions=[(-1.0, 1.0), (-1.0, 1.0)], base_estimator="GP",
        random_state=0)
    opt.init(2)
    params = np.empty(2)
    for _ in range(10):
        opt.get_next_parameters(params)
        feedback = [-np.linalg.norm(params - 0.5384 * np.ones(2))]
        opt.set_evaluation_feedback(feedback)
    assert_greater(opt.get_best_fitness(), -0.3)
    assert_equal(opt.get_best_fitness(),
                 -np.linalg.norm(opt.get_best_parameters() -
                                 0.5384 * np.ones(2)))
    assert_false(opt.is_behavior_learning_done())


def test_bayes_opt_gpr_minimize():
    opt = SkOptOptimizer(
        dimensions=[(-1.0, 1.0), (-1.0, 1.0)], base_estimator="GP",
        random_state=0, maximize=False)
    opt.init(2)
    params = np.empty(2)
    for _ in range(10):
        opt.get_next_parameters(params)
        feedback = [np.linalg.norm(params - 0.5384 * np.ones(2))]
        opt.set_evaluation_feedback(feedback)
    assert_greater(0.3, opt.get_best_fitness())


def test_bayes_opt_gbrt():
    opt = SkOptOptimizer(
        dimensions=[(-1.0, 1.0), (-1.0, 1.0)], base_estimator="GBRT",
        acq_optimizer="sampling", random_state=0)
    opt.init(2)
    params = np.empty(2)
    for _ in range(10):
        opt.get_next_parameters(params)
        feedback = [-np.linalg.norm(params - 0.5384 * np.ones(2))]
        opt.set_evaluation_feedback(feedback)
    assert_greater(opt.get_best_fitness(), -0.3)


def test_bayes_opt_rf():
    opt = SkOptOptimizer(
        dimensions=[(-1.0, 1.0), (-1.0, 1.0)], base_estimator="RF",
        acq_optimizer="sampling", random_state=0)
    opt.init(2)
    params = np.empty(2)
    for _ in range(10):
        opt.get_next_parameters(params)
        feedback = [-np.linalg.norm(params - 0.5384 * np.ones(2))]
        opt.set_evaluation_feedback(feedback)
    assert_greater(opt.get_best_fitness(), -0.3)


def test_bayes_opt_et():
    opt = SkOptOptimizer(
        dimensions=[(-1.0, 1.0), (-1.0, 1.0)], base_estimator="ET",
        acq_optimizer="sampling", random_state=0)
    opt.init(2)
    params = np.empty(2)
    for _ in range(10):
        opt.get_next_parameters(params)
        feedback = [-np.linalg.norm(params - 0.5384 * np.ones(2))]
        opt.set_evaluation_feedback(feedback)
    assert_greater(opt.get_best_fitness(), -0.3)


def test_bayes_opt_unknown_base_estimator():
    assert_raises_regexp(ValueError, "base_estimator parameter",
        SkOptOptimizer, dimensions=[(-1.0, 1.0), (-1.0, 1.0)],
        base_estimator="unknown", random_state=0)


def test_bayes_opt_wrong_dimensions():
    opt = SkOptOptimizer(
        dimensions=[(-1.0, 1.0), (-1.0, 1.0)], base_estimator="GP",
        random_state=0)
    assert_raises_regexp(ValueError, "Number of dimensions", opt.init, 1)
