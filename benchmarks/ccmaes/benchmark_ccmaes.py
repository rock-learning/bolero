import numpy as np
from bolero.optimizer import CCMAESOptimizer, CREPSOptimizer
from bolero.environment.contextual_objective_functions import \
    ContextualObjectiveFunction
from bolero.environment.objective_functions import rosenbrock


class Sphere(ContextualObjectiveFunction):
    def __init__(self, random_state, n_dims, n_context_dims):
        self.G = random_state.randn(n_dims, n_context_dims)

    def feedback(self, x, s):
        x2 = x + self.G.dot(s)
        return -x2.dot(x2)


class Rosenbrock(ContextualObjectiveFunction):
    def __init__(self, random_state, n_dims, n_context_dims):
        self.G = random_state.randn(n_dims, n_context_dims)

    def feedback(self, x, s):
        x2 = x + self.G.dot(s)
        return -rosenbrock(x2)  # TODO check if implementation is different from paper


def benchmark():
    objective_functions = {
        "sphere" : Sphere,
        "rosenbrock": Rosenbrock
    }
    algorithms = {
        "C-CMA-ES": CCMAESOptimizer,
        "C-REPS": CREPSOptimizer
    }
    seeds = list(range(1))#list(range(20)) TODO

    n_samples_per_update = 50
    additional_ctor_args = {
        "C-CMA-ES": {},
        "C-REPS": {
            "train_freq": n_samples_per_update,
            "epsilon": 1.0
        }
    }
    n_params = 20
    context_dims_per_objective = {
        "sphere": 2,
        "rosenbrock": 1
    }
    n_episodes = 1000

    # TODO use joblib to parallelize
    results = dict(
        (algorithm_name,
         dict((objective_name, np.empty((len(seeds), n_episodes)))
              for objective_name in objective_functions.keys()))
        for algorithm_name in algorithms.keys()
    )
    for objective_name, objective_function in objective_functions.items():
        for algorithm_name, algorithm in algorithms.items():
            for run_idx, seed in enumerate(seeds):
                n_context_dims = context_dims_per_objective[objective_name]
                random_state = np.random.RandomState(seed)
                # contexts are sampled uniformly from 1 <= s <= 2 (here: < 2)
                contexts = random_state.rand(n_episodes, n_context_dims) + 1.0
                obj = objective_function(random_state, n_params, n_context_dims)
                initial_params = random_state.randn(n_params)
                opt = algorithm(
                    initial_params=initial_params,
                    covariance=np.eye(n_params),
                    variance=1.0,
                    n_samples_per_update=n_samples_per_update,
                    **additional_ctor_args[algorithm_name]
                )
                opt.init(n_params, n_context_dims)

                feedbacks = np.empty(n_episodes)
                params = np.empty(n_params)
                for episode_idx in range(n_episodes):
                    opt.set_context(contexts[episode_idx])
                    opt.get_next_parameters(params)
                    feedbacks[episode_idx] = obj.feedback(
                        params, contexts[episode_idx])
                    opt.set_evaluation_feedback(
                        np.array([feedbacks[episode_idx]]))

                results[algorithm_name][objective_name][run_idx, :] = feedbacks

    import matplotlib.pyplot as plt
    for algorithm_name, algorithm_results in results.items():
        plt.plot(algorithm_results["rosenbrock"][0], label=algorithm_name)
    plt.xlabel("Episodes")
    plt.ylabel("Average Return")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    benchmark()