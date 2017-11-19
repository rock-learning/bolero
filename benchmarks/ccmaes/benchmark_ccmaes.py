import numpy as np
from bolero.optimizer import CCMAESOptimizer, CREPSOptimizer
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


def benchmark():
    objective_functions = {
        "sphere" : Sphere,
        #"rosenbrock": Rosenbrock
    }
    algorithms = {
        "C-CMA-ES": CCMAESOptimizer,
        #"C-REPS": CREPSOptimizer
    }
    seeds = list(range(1))#list(range(20)) TODO

    n_samples_per_update = 50
    additional_ctor_args = {
        "C-CMA-ES": {},
        "C-REPS": {
            "train_freq": n_samples_per_update,
            "epsilon": 1.0,
            "min_eta": 1e-10  # 1e-20 in the original code
        }
    }
    n_params = 20
    context_dims_per_objective = {
        "sphere": 2,
        "rosenbrock": 1
    }
    n_episodes = 200 * n_samples_per_update # 200 for Sphere, 850 for Rosenbrock

    # TODO use joblib to parallelize
    results = dict(
        (objective_name,
         dict((algorithm_name, np.empty((len(seeds), n_episodes)))
              for algorithm_name in algorithms.keys()))
        for objective_name in objective_functions.keys()
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
                    context_features="affine",
                    random_state=random_state,
                    gamma=1e-10,
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
                    opt.set_evaluation_feedback(feedbacks[episode_idx])

                results[objective_name][algorithm_name][run_idx, :] = feedbacks

    import matplotlib.pyplot as plt
    for objective_name, objective_results in results.items():
        plt.figure()
        plt.title(objective_name)
        for algorithm_name, algorithm_results in objective_results.items():
            feedbacks = algorithm_results[0]
            average_feedbacks = np.array(feedbacks).reshape(
                n_episodes / n_samples_per_update, n_samples_per_update).sum(axis=1)
            plt.plot(average_feedbacks, label=algorithm_name)
            plt.xlabel("Episodes")
            plt.ylabel("Average Return")
            plt.legend()
    plt.show()


if __name__ == "__main__":
    benchmark()