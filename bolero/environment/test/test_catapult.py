import numpy as np
from scipy.stats import uniform
from bolero.environment import Catapult
from nose.tools import (assert_equal, assert_less, assert_almost_equal,
                        assert_less_equal, assert_true, assert_raises_regexp)


def test_sample_contexts():
    env = Catapult(segments=[(0, 0), (20, 0)], context_interval=(0, 20),
                   random_state=0)
    env.init()
    assert_equal(env.get_num_context_dims(), 1)
    context = env.request_context(None)
    assert_less_equal(0.0, context[0])
    assert_less_equal(context[0], 1.0)


def test_sample_contexts_from_distribution():
    env = Catapult(segments=[(0, 0), (20, 0)], context_interval=(0, 20),
                   context_distribution=uniform(5, 10), random_state=0)
    env.init()

    contexts = np.empty(1000)
    for i in range(contexts.shape[0]):
        context = env.request_context(None)
        contexts[i] = context[0]

    norm_dist = uniform(0.25, 0.5)
    assert_true(np.all(0.25 <= contexts))
    assert_true(np.all(contexts <= 0.75))
    mean, var = norm_dist.stats("mv")
    assert_almost_equal(np.mean(contexts), mean, places=1)
    assert_almost_equal(np.var(contexts), var, places=1)


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


def test_intersection_fails():
    env = Catapult(segments=[(0, 0), (5, 0)], context_interval=(0, 10),
                   random_state=0)
    env.init()
    env.reset()
    env.set_inputs(np.array([10.0, np.deg2rad(45)]))
    assert_raises_regexp(
        Exception, "Could not intersect trajectory and surface",
        env.step_action)
