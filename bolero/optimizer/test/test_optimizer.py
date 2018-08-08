import numpy as np
from bolero.utils.testing import all_subclasses
from bolero.optimizer import Optimizer, ContextualOptimizer
from bolero.utils.testing import assert_pickle
from nose.tools import (assert_false, assert_true, assert_equal,
                        assert_raises_regexp)
from numpy.testing import assert_array_equal


ALL_OPTIMIZERS = all_subclasses(Optimizer)
ALL_CONTEXTUALOPTIMIZERS = all_subclasses(ContextualOptimizer)


def test_abstract_optimizer():
    class OptimizerSubclass(Optimizer):
        pass
    assert_raises_regexp(TypeError, "abstract class", OptimizerSubclass)


def test_abstract_contextual_optimizer():
    class OptimizerSubclass(ContextualOptimizer):
        pass
    assert_raises_regexp(TypeError, "abstract class", OptimizerSubclass)


def test_optimizers_have_default_constructor():
    for name, Optimizer in ALL_OPTIMIZERS:
        try:
            opt = Optimizer()
        except:
            raise AssertionError("Optimizer '%s' is not default "
                                 "constructable" % name)


def test_optimizers_follow_standard_protocol():
    for name, Optimizer in ALL_OPTIMIZERS:
        opt = Optimizer()
        n_params = 2
        opt.init(n_params)
        assert_false(opt.is_behavior_learning_done())
        params = np.empty(n_params)
        opt.get_next_parameters(params)
        assert_true(np.isfinite(params).all())
        opt.set_evaluation_feedback(np.array([0.0]))
        params = opt.get_best_parameters()
        assert_equal(len(params), n_params)
        assert_true(np.isfinite(params).all())

        assert_pickle(name, opt)


def test_contextual_optimizers_have_default_constructor():
    for name, ContextualOptimizer in ALL_CONTEXTUALOPTIMIZERS:
        try:
            opt = ContextualOptimizer()
        except:
            raise AssertionError("ContextualOptimizer '%s' is not default "
                                 "constructable" % name)


def test_contextual_optimizers_follow_standard_protocol():
    for name, ContextualOptimizer in ALL_CONTEXTUALOPTIMIZERS:
        opt = ContextualOptimizer()
        n_params = 1
        n_context_dims = 1
        opt.init(n_params, n_context_dims)
        context = opt.get_desired_context()
        if context is None:
            context = np.zeros(n_context_dims)
        opt.set_context(context)
        assert_false(opt.is_behavior_learning_done())
        params = np.empty(n_params)
        opt.get_next_parameters(params)
        assert_true(np.isfinite(params).all())
        opt.set_evaluation_feedback(np.array([0.0]))

        policy = opt.best_policy()
        assert_true(np.isfinite(policy(context)).all())

        assert_pickle(name, opt)
