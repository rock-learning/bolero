import numpy as np
from bolero.representation import (ConstantBehavior, DummyBehavior,
                                   RandomBehavior)
from bolero.utils.testing import assert_raise_message
from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal


def test_constant_behavior():
    const = np.array([1.3, 2.2])
    beh = ConstantBehavior(3, 2, const)

    assert_equal(beh.get_n_params(), 0)
    assert_array_equal(beh.get_params(), np.array([]))

    outputs = np.empty(2)
    for _ in range(100):
        beh.get_outputs(outputs)
        assert_array_equal(outputs, const)

    beh = ConstantBehavior(3, 2)
    beh.get_outputs(outputs)
    assert_array_equal(outputs, np.zeros(2))


def test_dummy_behavior():
    params = np.array([1.4, 2.3])
    beh = DummyBehavior(initial_params=params)

    assert_equal(beh.get_n_params(), 2)
    assert_array_equal(beh.get_params(), params)

    outputs = np.empty(2)
    beh.get_outputs(outputs)
    assert_array_equal(outputs, params)

    beh = DummyBehavior()
    assert_raise_message(ValueError, "parameters have not been set",
                         beh.get_n_params)
    assert_raise_message(ValueError, "parameters have not been set",
                         beh.get_params)

    beh.set_params(params)
    assert_equal(beh.get_n_params(), 2)
    assert_array_equal(beh.get_params(), params)


def test_random_behavior():
    beh = RandomBehavior(4, 5, random_state=0)

    assert_equal(beh.get_n_params(), 0)
    assert_array_equal(beh.get_params(), np.array([]))

    outputs = np.empty(5)
    outputs[:] = np.nan
    beh.get_outputs(outputs)
    assert_true(np.isfinite(outputs).all())
