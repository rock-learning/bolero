import numpy as np
from bolero.representation import LinearBehavior
from numpy.testing import assert_almost_equal


def test_zero_weights():
    beh = LinearBehavior()
    beh.init(1, 1)
    beh.set_params(np.zeros(2))

    inputs = np.ones(1)
    outputs = np.zeros(1)
    beh.set_inputs(inputs)
    beh.step()
    beh.get_outputs(outputs)
    assert_almost_equal(outputs[0], 0.0)


def test_one_weights():
    beh = LinearBehavior()
    beh.init(1, 1)
    beh.set_params(np.ones(2))

    inputs = np.ones(1)
    outputs = np.zeros(1)
    beh.set_inputs(inputs)
    beh.step()
    beh.get_outputs(outputs)
    assert_almost_equal(outputs[0], 2.0)
