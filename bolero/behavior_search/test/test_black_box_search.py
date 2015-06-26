import numpy as np
import os
from bolero.behavior_search import BlackBoxSearch
from bolero.representation import ConstantBehavior
from bolero.optimizer import NoOptimizer
from bolero.utils.testing import assert_pickle
from nose.tools import assert_false, assert_true, assert_raises_regexp
from numpy.testing import assert_array_equal


def test_black_box_search_requires_optimizer():
    class NoOptimizerSubclass(object):
        pass

    bs = BlackBoxSearch(ConstantBehavior(), NoOptimizerSubclass())
    assert_raises_regexp(TypeError, "expects instance of Optimizer",
                         bs.init, 5, 5)


def test_black_box_search_from_dicts():
    beh = {"type": "bolero.representation.ConstantBehavior"}
    opt = {"type": "bolero.optimizer.NoOptimizer"}
    bs = BlackBoxSearch(beh, opt)
    bs.init(5, 5)
    # NoOptimizer should be initialized with the parameters from the behavior
    assert_array_equal(bs.behavior.get_params(), bs.optimizer.initial_params)


def test_black_box_search_protocol():
    n_inputs, n_outputs = 5, 5

    bs = BlackBoxSearch(ConstantBehavior(), NoOptimizer())
    bs.init(n_inputs, n_outputs)
    assert_false(bs.is_behavior_learning_done())

    beh = bs.get_next_behavior()

    inputs = np.zeros(n_inputs)
    beh.set_inputs(inputs)
    outputs = np.empty(n_outputs)
    beh.get_outputs(outputs)

    bs.set_evaluation_feedback(np.array([0.0]))


def test_save_black_box_search():
    bs = BlackBoxSearch(ConstantBehavior(), NoOptimizer())
    bs.init(5, 5)

    assert_pickle("BlackBoxSearch", bs)

    path = "." + os.sep
    bs.write_results(path)
    bs.get_behavior_from_results(path)
    filename = path + "BlackBoxSearch.pickle"
    assert_true(os.path.exists(filename))
    if os.path.exists(filename):
        os.remove(filename)
