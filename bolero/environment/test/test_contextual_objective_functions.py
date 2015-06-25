import numpy as np
from bolero.environment.contextual_objective_functions import (
    CONTEXTUAL_FUNCTIONS, ContextualObjectiveFunction)
from bolero.utils.testing import assert_raise_message
from nose.tools import assert_less, assert_almost_equal


def test_optimum():
    random_state = np.random.RandomState(0)
    s = np.zeros(1)
    for name, Objective in CONTEXTUAL_FUNCTIONS.items():
        objective = Objective(random_state, n_dims=1, n_context_dims=1)
        x_opt = objective.x_opt(s)
        f_opt = objective.feedback(x_opt, s)
        f = objective.feedback(x_opt + random_state.randn(2), s)
        assert_less(f, f_opt)
        assert_almost_equal(f_opt, objective.f_opt(s),
                            msg="Optimum %g of '%s' is not optimal (%g)"
                            % (f_opt, name, objective.f_opt(s)))


def test_input_validation():
    env = ContextualObjectiveFunction("Unknown", 2)
    assert_raise_message(ValueError, "Unknown function", env.init)
    env = ContextualObjectiveFunction("LinearContextualSphere", 0, 1)
    assert_raise_message(ValueError, "Number of parameters", env.init)
    env = ContextualObjectiveFunction("LinearContextualSphere", 1, 0)
    assert_raise_message(ValueError, "Number of context dimensions", env.init)
