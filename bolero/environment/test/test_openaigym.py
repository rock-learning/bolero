import numpy as np
from bolero.environment import OpenAiGym, gym_available
if not gym_available:
    from nose import SkipTest
    raise SkipTest("gym is not installed")
from nose.tools import assert_equal, assert_true


def test_discrete_input():
    env = OpenAiGym("CartPole-v0")
    env.init()

    assert_equal(env.get_num_inputs(), 1)
    assert_equal(env.get_num_outputs(), 4)

    inputs = np.zeros(env.get_num_inputs())
    outputs = np.zeros(env.get_num_outputs())

    env.reset()
    env.get_outputs(outputs)
    i = 0
    while not env.is_evaluation_done():
        env.set_inputs(inputs)
        env.step_action()
        env.get_outputs(outputs)
        i += 1

    assert_true(env.is_evaluation_done())

    feedback = env.get_feedback()
    assert_equal(len(feedback), i)

    assert_equal(env.get_maximum_feedback(), 195.0)


def test_box_input():
    env = OpenAiGym("Pendulum-v0")
    env.init()

    assert_equal(env.get_num_inputs(), 1)
    assert_equal(env.get_num_outputs(), 3)

    inputs = np.zeros(env.get_num_inputs())
    outputs = np.zeros(env.get_num_outputs())

    env.reset()
    env.get_outputs(outputs)
    i = 0
    while not env.is_evaluation_done():
        env.set_inputs(inputs)
        env.step_action()
        env.get_outputs(outputs)
        i += 1

    assert_true(env.is_evaluation_done())

    feedback = env.get_feedback()
    assert_equal(len(feedback), i)

    assert_equal(env.get_maximum_feedback(), float("inf"))


def test_tuple_output():
    env = OpenAiGym("Blackjack-v0")
    env.init()

    assert_equal(env.get_num_inputs(), 1)
    assert_equal(env.get_num_outputs(), 3)

    inputs = np.zeros(env.get_num_inputs())
    outputs = np.zeros(env.get_num_outputs())

    env.reset()
    env.get_outputs(outputs)
    i = 0
    while not env.is_evaluation_done():
        env.set_inputs(inputs)
        env.step_action()
        env.get_outputs(outputs)
        i += 1

    assert_true(env.is_evaluation_done())

    feedback = env.get_feedback()
    assert_equal(len(feedback), i)

    assert_equal(env.get_maximum_feedback(), float("inf"))


def test_tuple_input():
    env = OpenAiGym("Copy-v0")
    env.init()

    assert_equal(env.get_num_inputs(), 3)
    assert_equal(env.get_num_outputs(), 1)

    inputs = np.ones(env.get_num_inputs())
    outputs = np.ones(env.get_num_outputs())

    env.reset()
    env.get_outputs(outputs)
    i = 0
    while not env.is_evaluation_done():
        env.set_inputs(inputs)
        env.step_action()
        env.get_outputs(outputs)
        i += 1

    assert_true(env.is_evaluation_done())

    feedback = env.get_feedback()
    assert_equal(len(feedback), i)

    assert_equal(env.get_maximum_feedback(), 25.0)
