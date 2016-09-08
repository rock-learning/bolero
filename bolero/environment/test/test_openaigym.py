import numpy as np
from bolero.environment import OpenAiGym
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


def test_box_input():
    env = OpenAiGym("InvertedPendulum-v1")
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
