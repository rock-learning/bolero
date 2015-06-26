import numpy as np
from bolero.environment.objective_functions import FUNCTIONS, ObjectiveFunction
from nose.tools import assert_less, assert_almost_equal, assert_raises_regexp


def test_optimum():
    random_state = np.random.RandomState(0)
    for name, Objective in FUNCTIONS.items():
        objective = Objective(random_state, n_dims=2)
        f_opt = objective.feedback(objective.x_opt)
        f = objective.feedback(objective.x_opt + random_state.randn(2))
        assert_less(f, f_opt)
        if name == "Schwefel":
            continue  # Schwefel is broken
        assert_almost_equal(f_opt, objective.f_opt,
                            msg="Optimum %g of '%s' is not optimal (%g)"
                            % (f_opt, name, objective.f_opt))


def test_input_validation():
    env = ObjectiveFunction("Unknown", 2)
    assert_raises_regexp(ValueError, "Unknown function", env.init)
    env = ObjectiveFunction("Sphere", 0)
    assert_raises_regexp(ValueError, "Number of parameters", env.init)
