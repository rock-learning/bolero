import numpy as np
from bolero.utils.testing import all_subclasses
from bolero.environment import Environment, ContextualEnvironment
from nose.tools import (assert_false, assert_true, assert_greater,
                        assert_greater_equal)


ALL_ENVIRONMENTS = all_subclasses(Environment, ["SetContext"])


def test_environments_have_default_constructor():
    for name, Environment in ALL_ENVIRONMENTS:
        try:
            env = Environment()
        except:
            raise AssertionError("Environment '%s' is not default "
                                 "constructable" % name)


def test_environments_follow_standard_protocol():
    for _, Environment in ALL_ENVIRONMENTS:
        env = Environment()
        env.init()

        n_inputs = env.get_num_inputs()
        assert_greater(n_inputs, 0)
        n_outputs = env.get_num_outputs()
        assert_greater_equal(n_outputs, 0)

        inputs = np.zeros(n_inputs)
        outputs = np.empty(n_outputs)
        outputs[:] = np.nan

        i = 0
        env.reset()
        while not env.is_evaluation_done():
            env.get_outputs(outputs)
            env.set_inputs(inputs)
            env.step_action()

            i += 1
            if i >= 1000:
                break
        assert_true(env.is_evaluation_done())
        assert_true(np.isfinite(outputs).all())

        feedback = env.get_feedback()
        assert_false(env.is_behavior_learning_done())
        if not isinstance(env, ContextualEnvironment):
            assert_greater_equal(env.get_maximum_feedback(), np.sum(feedback))
