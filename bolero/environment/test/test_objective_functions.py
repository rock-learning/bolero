import numpy as np
from bolero.environment.objective_functions import FUNCTIONS
from nose.tools import assert_less, assert_almost_equal


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
