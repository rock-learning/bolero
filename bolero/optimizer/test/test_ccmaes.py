import numpy as np
from bolero.environment.contextual_objective_functions import \
    LinearContextualSphere
from bolero.optimizer import CCMAESOptimizer
#from bolero.optimizer import CREPSOptimizer


def test_linear_contextual_sphere():
    random_state = np.random.RandomState(0)
    n_params = 3
    n_context_dims = 2
    obj = LinearContextualSphere(random_state, n_params, n_context_dims)

    opt = CCMAESOptimizer(context_features="affine", random_state=random_state)
    #opt = CREPSOptimizer(context_features="affine", random_state=random_state)
    opt.init(n_params, n_context_dims)
    params = np.empty(n_params)
    for i in range(1000):
        context = random_state.rand(n_context_dims) * 2.0 - 1.0
        opt.set_context(context)
        opt.get_next_parameters(params)
        opt.set_evaluation_feedback([obj.feedback(params, context)])

    policy = opt.best_policy()
    c1 = c2 = np.linspace(-1, 1, 11)
    C1, C2 = np.meshgrid(c1, c2)
    test_contexts = np.array(zip(C1.ravel(), C2.ravel()))
    f = np.array([obj.feedback(policy(s), s) for s in test_contexts])
    f_opt = np.array([obj.f_opt(np.array(s)) for s in test_contexts])
    print(np.linalg.norm(f - f_opt))