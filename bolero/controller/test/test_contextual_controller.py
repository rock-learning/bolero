import numpy as np
from nose.tools import (assert_equal, assert_less, assert_greater, assert_true,
                        assert_raises_regexp)
from bolero.controller import ContextualController
from bolero.environment import ObjectiveFunction, ContextualObjectiveFunction
from bolero.behavior_search import JustOptimizer, JustContextualOptimizer
from bolero.representation import DummyBehavior
from bolero.optimizer import CMAESOptimizer, CREPSOptimizer
from numpy.testing import assert_array_equal


def test_missing_environment():
    assert_raises_regexp(ValueError, "Environment specification is missing",
                         ContextualController)


def test_noncontextual_environment():
    assert_raises_regexp(TypeError, "requires contextual environment",
                         ContextualController, environment=ObjectiveFunction())


def test_noncontextual_behavior_search():
    opt = CMAESOptimizer(initial_params=np.zeros(1))
    assert_raises_regexp(
        TypeError, "requires contextual behavior search", ContextualController,
        environment=ContextualObjectiveFunction(),
        behavior_search=JustOptimizer(opt))


def test_missing_behavior_search():
    ctrl = ContextualController(environment=ContextualObjectiveFunction())
    beh = DummyBehavior(initial_params=np.array([0.0]))
    beh.init(0, 1)
    feedback = ctrl.episode_with(beh)
    assert_equal(len(feedback), 1)
    assert_less(feedback[0], ctrl.environment.get_maximum_feedback(
        ctrl.environment.context))


def test_learning_fails_with_missing_behavior_search():
    controller = ContextualController(environment=ContextualObjectiveFunction())
    assert_raises_regexp(ValueError, "BehaviorSearch is required",
                         controller.learn)


def test_controller_creps_objective():
    opt = CREPSOptimizer(initial_params=np.zeros(1))
    ctrl = ContextualController(environment=ContextualObjectiveFunction(),
                                behavior_search=JustContextualOptimizer(opt))
    returns = ctrl.learn()
    assert_equal(len(returns), 10)


def test_controller_cmaes_sphere_via_config():
    config = {
        "Environment": {"type": "bolero.environment.ContextualObjectiveFunction"},
        "BehaviorSearch": {
            "type": "bolero.behavior_search.JustContextualOptimizer",
            "optimizer": {"type": "bolero.optimizer.CREPSOptimizer",
                          "initial_params": np.zeros(1)}}
    }
    ctrl = ContextualController(config)
    returns = ctrl.learn()
    assert_equal(len(returns), 10)


def test_record_inputs():
    opt = CREPSOptimizer(initial_params=np.zeros(1))
    ctrl = ContextualController(environment=ContextualObjectiveFunction(),
                                behavior_search=JustContextualOptimizer(opt),
                                record_inputs=True)
    returns = ctrl.learn()
    assert_equal(len(returns), 10)
    assert_equal(np.array(ctrl.inputs_).shape, (10, 1, 1))


def test_record_outputs():
    opt = CREPSOptimizer(initial_params=np.zeros(1))
    ctrl = ContextualController(environment=ContextualObjectiveFunction(),
                                behavior_search=JustContextualOptimizer(opt),
                                record_outputs=True)
    returns = ctrl.learn()
    assert_equal(len(returns), 10)
    assert_equal(np.array(ctrl.outputs_).shape, (10, 1, 0))


def test_record_feedbacks():
    opt = CREPSOptimizer(initial_params=np.zeros(1))
    ctrl = ContextualController(environment=ContextualObjectiveFunction(),
                                behavior_search=JustContextualOptimizer(opt),
                                record_feedbacks=True)
    returns = ctrl.learn()
    assert_array_equal(returns, ctrl.feedbacks_)


def test_learn_controller_cmaes_sphere():
    test_contexts = np.linspace(-5, 5, 11)[:, np.newaxis]

    opt = CREPSOptimizer(initial_params=np.zeros(1), random_state=0)
    ctrl = ContextualController(
        environment=ContextualObjectiveFunction(),
        behavior_search=JustContextualOptimizer(opt),
        n_episodes=200, n_episodes_before_test=200, test_contexts=test_contexts)
    ctrl.learn()
    for d in ctrl.test_results_[-1]:
        assert_greater(d, -1e-5)


def test_record_contexts():
    contexts = np.linspace(-5, 5, 11)[:, np.newaxis]

    class CREPSFixedContextOrder(CREPSOptimizer):
        def get_desired_context(self):
            return contexts[self.it]

    opt = CREPSFixedContextOrder()
    ctrl = ContextualController(
        environment=ContextualObjectiveFunction(),
        behavior_search=JustContextualOptimizer(opt),
        n_episodes=11, record_contexts=True)
    ctrl.learn()
    assert_array_equal(contexts, ctrl.contexts_)


def test_context_cannot_be_set():
    class EnvironmentWithRandomContext(ContextualObjectiveFunction):
        def request_context(self, _):
            self.context = np.random.randn(self.n_context_dims)
            return self.context

    test_contexts = np.linspace(-5, 5, 11)[:, np.newaxis]

    opt = CREPSOptimizer(initial_params=np.zeros(1), random_state=0)
    ctrl = ContextualController(
        environment=EnvironmentWithRandomContext(),
        behavior_search=JustContextualOptimizer(opt),
        n_episodes=2, n_episodes_before_test=1, test_contexts=test_contexts)
    assert_raises_regexp(Exception, "could not set context", ctrl.learn)
