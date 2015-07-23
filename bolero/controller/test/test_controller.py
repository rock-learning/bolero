import numpy as np
from nose.tools import (assert_equal, assert_less, assert_greater, assert_true,
                        assert_raises_regexp)
from bolero.controller import Controller
from bolero.environment import ObjectiveFunction
from bolero.behavior_search import JustOptimizer
from bolero.representation import DummyBehavior
from bolero.optimizer import CMAESOptimizer
from numpy.testing import assert_array_equal


def test_missing_environment():
    assert_raises_regexp(ValueError, "Environment specification is missing",
                        Controller)


def test_no_environment_subclass():
    class NoEnvironment(object):
        pass

    assert_raises_regexp(
        TypeError, "requires subclass of 'Environment'",
        Controller, environment=NoEnvironment())

def test_no_behavior_search_subclass():
    class NoBehaviorSearch(object):
        pass

    assert_raises_regexp(
        TypeError, "requires subclass of 'BehaviorSearch'",
        Controller, environment=ObjectiveFunction(),
        behavior_search=NoBehaviorSearch())


def test_missing_behavior_search():
    ctrl = Controller(environment=ObjectiveFunction())
    beh = DummyBehavior(initial_params=np.array([0.0, 0.0]))
    beh.init(0, 2)
    feedback = ctrl.episode_with(beh)
    assert_equal(len(feedback), 1)
    assert_less(feedback[0], ctrl.environment.get_maximum_feedback())


def test_learning_fails_with_missing_behavior_search():
    controller = Controller(environment=ObjectiveFunction())
    assert_raises_regexp(ValueError, "BehaviorSearch is required",
                        controller.learn)


def test_controller_cmaes_sphere():
    opt = CMAESOptimizer(initial_params=np.zeros(2))
    ctrl = Controller(environment=ObjectiveFunction(),
                      behavior_search=JustOptimizer(opt))
    returns = ctrl.learn()
    assert_equal(len(returns), 10)


def test_controller_cmaes_sphere_via_config():
    config = {
        "Environment": {"type": "bolero.environment.ObjectiveFunction"},
        "BehaviorSearch": {
            "type": "bolero.behavior_search.JustOptimizer",
            "optimizer": {"type": "bolero.optimizer.CMAESOptimizer",
                          "initial_params": np.zeros(2)}}
    }
    ctrl = Controller(config)
    returns = ctrl.learn()
    assert_equal(len(returns), 10)


def test_record_trajectories():
    opt = CMAESOptimizer(initial_params=np.zeros(2))
    ctrl = Controller(environment=ObjectiveFunction(),
                      behavior_search=JustOptimizer(opt),
                      record_trajectories=True)
    returns = ctrl.learn()
    assert_equal(len(returns), 10)
    assert_equal(np.array(ctrl.trajectories_).shape, (10, 1, 2))


def test_record_feedbacks():
    opt = CMAESOptimizer(initial_params=np.zeros(2))
    ctrl = Controller(environment=ObjectiveFunction(),
                      behavior_search=JustOptimizer(opt),
                      record_feedbacks=True)
    returns = ctrl.learn()
    assert_array_equal(returns, ctrl.feedbacks_)


def test_learn_controller_cmaes_sphere():
    opt = CMAESOptimizer(initial_params=np.zeros(2), random_state=0)
    ctrl = Controller(environment=ObjectiveFunction(random_state=0),
                      behavior_search=JustOptimizer(opt),
                      n_episodes=200)
    returns = ctrl.learn()
    dist_to_maximum = returns.max() - ctrl.environment.get_maximum_feedback()
    assert_greater(dist_to_maximum, -1e-5)
