import numpy as np
import os
from bolero.behavior_search import BlackBoxSearch
from bolero.representation import ConstantBehavior
from bolero.optimizer import NoOptimizer
from bolero.utils.testing import assert_pickle
from nose.tools import assert_false, assert_true


def test_black_box_search():
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

    assert_pickle("BlackBoxSearch", bs)

    path = "." + os.sep
    bs.write_results(path)
    bs.get_behavior_from_results(path)
    filename = path + "BlackBoxSearch.pickle"
    assert_true(os.path.exists(filename))
    if os.path.exists(filename):
        os.remove(filename)
