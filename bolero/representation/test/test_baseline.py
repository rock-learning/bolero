import numpy as np
from bolero.representation import (ConstantBehavior, DummyBehavior,
                                   RandomBehavior)
from nose.tools import assert_equal, assert_true, assert_raises_regexp
from numpy.testing import assert_array_equal


def test_dummy_behavior():
    beh = DummyBehavior()
    beh.init(0, 2)
    assert_raises_regexp(ValueError, "parameters have not been set",
                         beh.get_params)

    params = np.array([1.4, 2.3])
    beh = DummyBehavior(initial_params=params)
    beh.init(0, 2)

    assert_equal(beh.get_n_params(), 2)
    assert_array_equal(beh.get_params(), params)

    outputs = np.empty(2)
    beh.get_outputs(outputs)
    assert_array_equal(outputs, params)

    beh.set_params(params)
    assert_equal(beh.get_n_params(), 2)
    assert_array_equal(beh.get_params(), params)

    assert_raises_regexp(
        NotImplementedError, "does not accept any meta parameters",
        beh.set_meta_parameters, ["key"], [0.0])
    beh.reset()


def test_constant_behavior():
    const = np.array([1.3, 2.2])
    beh = ConstantBehavior(const)
    beh.init(3, 2)

    assert_equal(beh.get_n_params(), 0)
    assert_array_equal(beh.get_params(), np.array([]))

    outputs = np.empty(2)
    for _ in range(100):
        beh.get_outputs(outputs)
        assert_array_equal(outputs, const)

    beh = ConstantBehavior()
    beh.init(3, 2)
    beh.get_outputs(outputs)
    assert_array_equal(outputs, np.zeros(2))

    assert_raises_regexp(
        NotImplementedError, "does not accept any meta parameters",
        beh.set_meta_parameters, ["key"], [0.0])
    beh.reset()

    assert_raises_regexp(
        ValueError, "Length of parameter vector must be 0",
        beh.set_params, np.zeros(2))
    beh.set_params(np.array([]))


def test_random_behavior():
    beh = RandomBehavior(random_state=0)
    beh.init(4, 5)

    assert_equal(beh.get_n_params(), 0)
    assert_array_equal(beh.get_params(), np.array([]))

    outputs = np.empty(5)
    outputs[:] = np.nan
    beh.get_outputs(outputs)
    assert_true(np.isfinite(outputs).all())

    assert_raises_regexp(
        NotImplementedError, "does not accept any meta parameters",
        beh.set_meta_parameters, ["key"], [0.0])
    beh.reset()

    assert_raises_regexp(
        ValueError, "Length of parameter vector must be 0",
        beh.set_params, np.zeros(2))
    beh.set_params(np.array([]))
