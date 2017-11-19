import numpy as np
from bolero.environment.contextual_objective_functions import \
    LinearContextualSphere
from bolero.optimizer import CCMAESOptimizer
from nose.tools import assert_greater, assert_raises_regexp


def test_cmaes_no_initial_params():
    opt = CCMAESOptimizer()
    opt.init(10, 5)
    params = np.empty(10)
    opt.set_context(np.zeros(5))
    opt.get_next_parameters(params)


def test_cmaes_dimensions_mismatch():
    opt = CCMAESOptimizer(initial_params=np.zeros(5))
    assert_raises_regexp(ValueError, "Number of dimensions", opt.init, 10, 5)


def test_cmaes_diagonal_cov():
    opt = CCMAESOptimizer(covariance=np.zeros(10))
    opt.init(10, 5)
    params = np.empty(10)
    opt.set_context(np.zeros(5))
    opt.get_next_parameters(params)


def evaluate(policy, obj):
    c1 = c2 = np.linspace(-1, 1, 11)
    C1, C2 = np.meshgrid(c1, c2)
    test_contexts = np.array(zip(C1.ravel(), C2.ravel()))
    f = np.array([obj.feedback(policy(s, explore=False), s)
                  for s in test_contexts])
    f_opt = np.array([obj.f_opt(s) for s in test_contexts])
    return np.mean(f - f_opt)


def test_linear_contextual_sphere():
    random_state = np.random.RandomState(0)
    n_params = 3
    n_context_dims = 2
    obj = LinearContextualSphere(random_state, n_params, n_context_dims)

    opt = CCMAESOptimizer(context_features="affine", random_state=random_state,
                          log_to_stdout=False)
    opt.init(n_params, n_context_dims)
    params = np.empty(n_params)
    for i in range(1000):
        context = random_state.rand(n_context_dims) * 2.0 - 1.0
        opt.set_context(context)
        opt.get_next_parameters(params)
        opt.set_evaluation_feedback([obj.feedback(params, context)])
    policy = opt.best_policy()
    mean_reward = evaluate(policy, obj)
    assert_greater(mean_reward, -1e-5)
