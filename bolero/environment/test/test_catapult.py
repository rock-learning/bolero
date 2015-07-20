import numpy as np
from bolero.environment import Catapult
from nose.tools import (assert_equal, assert_less, assert_almost_equal,
                        assert_less_equal)


def test_sample_contexts():
    env = Catapult(segments=[(0, 0), (20, 0)], context_interval=(0, 20),
                   random_state=0)
    env.init()
    assert_equal(env.get_num_context_dims(), 1)
    context = env.request_context(None)
    assert_less_equal(0.0, context[0])
    assert_less_equal(context[0], 1.0)
    print(context[0])


def test_vary_angle():
    env = Catapult(segments=[(0, 0), (20, 0)], context_interval=(0, 20),
                   random_state=0)
    env.init()
    context = env.request_context(np.array([0.5]))
    assert_equal(context[0], 0.5)

    env.reset()
    env.set_inputs(np.array([10.0, np.deg2rad(90)]))
    env.step_action()
    reward_90deg = env.get_feedback()[0]
    assert_almost_equal(reward_90deg, -11.0)

    env.reset()
    env.set_inputs(np.array([10.0, np.deg2rad(45)]))
    env.step_action()
    reward_45deg = env.get_feedback()[0]

    env.reset()
    env.set_inputs(np.array([10.0, np.deg2rad(55)]))
    env.step_action()
    reward_55deg = env.get_feedback()[0]

    env.reset()
    env.set_inputs(np.array([10.0, np.deg2rad(35)]))
    env.step_action()
    reward_35deg = env.get_feedback()[0]

    assert_equal(reward_35deg, reward_55deg)
    assert_less(reward_35deg, reward_45deg)
    assert_less(reward_55deg, reward_45deg)

    max_feedback = env.get_maximum_feedback(context)
    assert_less(reward_45deg, max_feedback)


def test_vary_velocity():
    env = Catapult(segments=[(0, 0), (20, 0)], context_interval=(0, 20),
                   random_state=0)
    env.init()
    context = env.request_context(np.array([0.5]))
    assert_equal(context[0], 0.5)

    env.reset()
    env.set_inputs(np.array([3.0, np.deg2rad(45)]))
    env.step_action()
    reward_v3 = env.get_feedback()[0]

    env.reset()
    env.set_inputs(np.array([6.0, np.deg2rad(45)]))
    env.step_action()
    reward_v6 = env.get_feedback()[0]

    env.reset()
    env.set_inputs(np.array([9.0, np.deg2rad(45)]))
    env.step_action()
    reward_v9 = env.get_feedback()[0]

    assert_less(reward_v3, reward_v6)
    assert_less(reward_v6, reward_v9)

    max_feedback = env.get_maximum_feedback(context)
    assert_less(reward_v9, max_feedback)

    max_feedback = env.get_maximum_feedback(context)
    assert_less(reward_v9, max_feedback)
