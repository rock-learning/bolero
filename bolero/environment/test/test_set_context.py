import numpy as np
from bolero.environment import ObjectiveFunction, ContextualObjectiveFunction
from bolero.environment import SetContext
from nose.tools import (assert_almost_equal, assert_equal, assert_true,
                        assert_false)


def test_set_context():
    env = ObjectiveFunction(name="Sphere", n_params=2, random_state=0)
    env.init()

    cenv = ContextualObjectiveFunction(
        name="LinearContextualSphere", n_params=2, n_context_dims=1,
        random_state=0)
    env2 = SetContext(cenv, context=np.array([0.0]))
    env2.init()

    assert_equal(env.get_num_inputs(), env2.get_num_inputs())
    assert_equal(env.get_num_outputs(), env2.get_num_outputs())

    assert_almost_equal(env.get_maximum_feedback(), env2.get_maximum_feedback())

    params = np.zeros(2)
    f = np.empty(2)
    for i, e in enumerate([env, env2]):
        e.reset()
        e.set_inputs(params)
        e.step_action()
        out = np.empty(1)
        e.get_outputs(out)
        assert_true(e.is_evaluation_done())
        f[i] = e.get_feedback()[:]
        assert_false(e.is_behavior_learning_done())
    assert_almost_equal(f[0], f[1])
