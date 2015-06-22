import numpy as np
from bolero.utils.testing import all_subclasses
from bolero.optimizer import Optimizer
from nose.tools import assert_false, assert_true, assert_equal


ALL_OPTIMIZERS = all_subclasses(Optimizer)


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
