from bolero.environment.contextual_objective_functions import \
    ContextualObjectiveFunction
from bolero.environment.objective_functions import rosenbrock


class Sphere(ContextualObjectiveFunction):
    def __init__(self, random_state, n_dims, n_context_dims):
        self.G = random_state.randn(n_dims, n_context_dims)

    def feedback(self, theta, s):
        x = theta + self.G.dot(s)
        return -x.dot(x)


class Rosenbrock(ContextualObjectiveFunction):
    def __init__(self, random_state, n_dims, n_context_dims):
        self.G = random_state.randn(n_dims, n_context_dims)

    def feedback(self, theta, s):
        x = theta + self.G.dot(s)
        return -rosenbrock(x)

