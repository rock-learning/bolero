import numpy as np
from bolero.environment.contextual_objective_functions import \
    LinearContextualSphere, ConstantContextualSphere, QuadraticContextualSphere
from bolero.optimizer import CCMAESOptimizer
from bolero.optimizer import CREPSOptimizer


def evaluate(policy, obj):
    c1 = c2 = np.linspace(-1, 1, 11)
    C1, C2 = np.meshgrid(c1, c2)
    test_contexts = np.array(zip(C1.ravel(), C2.ravel()))
    f = np.array([obj.feedback(policy(s), s) for s in test_contexts])
    f_opt = np.array([obj.f_opt(np.array(s)) for s in test_contexts])
    return np.linalg.norm(f - f_opt)


def test_linear_contextual_sphere():
    random_state = np.random.RandomState(0)
    n_params = 3
    n_context_dims = 2
    obj = QuadraticContextualSphere(random_state, n_params, n_context_dims)
    #obj = LinearContextualSphere(random_state, n_params, n_context_dims)
    #obj = ConstantContextualSphere(random_state, n_params, n_context_dims)

    opt = CCMAESOptimizer(context_features="quadratic", random_state=random_state,
                          log_to_stdout=False)
    #opt = CREPSOptimizer(context_features="quadratic", random_state=random_state)
    opt.init(n_params, n_context_dims)
    params = np.empty(n_params)
    for i in range(10000):
        context = random_state.rand(n_context_dims) * 2.0 - 1.0
        opt.set_context(context)
        opt.get_next_parameters(params)
        opt.set_evaluation_feedback([obj.feedback(params, context)])
        if i % 50 == 0:
            policy = opt.best_policy()
            print(evaluate(policy, obj))
    # TODO compare learning plots of C-REPS and C-CMA-ES (and move to example)
