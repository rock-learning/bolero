import numpy as np
from joblib import Parallel, delayed
from bolero.optimizer import CCMAESOptimizer, CREPSOptimizer
from objective import Sphere, Rosenbrock


objective_functions = {
    "sphere" : Sphere,
    #"rosenbrock": Rosenbrock
}
algorithms = {
    "C-CMA-ES": CCMAESOptimizer,
    #"C-REPS": CREPSOptimizer
}
seeds = list(range(8))#list(range(20)) TODO

n_samples_per_update = 50
additional_ctor_args = {
    "C-CMA-ES": {
        "baseline_degree": 2
    },
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
n_episodes = 250 * n_samples_per_update # 200 for Sphere, 850 for Rosenbrock


def benchmark():
    results = dict(
        (objective_name,
         dict((algorithm_name, [])
              for algorithm_name in algorithms.keys()))
        for objective_name in objective_functions.keys()
    )
    for objective_name in objective_functions.keys():
        for algorithm_name in algorithms.keys():
            #for run_idx, seed in enumerate(seeds):
            #    feedbacks = optimize(objective_name, algorithm_name, seed)
            #    results[objective_name][algorithm_name].append(feedbacks)
            feedbacks = Parallel(n_jobs=8)(
                delayed(optimize)(objective_name, algorithm_name, seed)
                for run_idx, seed in enumerate(seeds))
            results[objective_name][algorithm_name] = feedbacks

    import matplotlib.pyplot as plt
    for objective_name, objective_results in results.items():
        plt.figure()
        plt.title(objective_name)
        for algorithm_name, algorithm_results in objective_results.items():
            n_generations = n_episodes / n_samples_per_update
            grouped_feedbacks = np.array(algorithm_results).reshape(
                len(seeds), n_generations, n_samples_per_update)
            average_feedback_per_generation = grouped_feedbacks.mean(axis=2)
            mean_feedbacks = average_feedback_per_generation.mean(axis=0)
            std_feedbacks = average_feedback_per_generation.std(axis=0)
            generation = np.arange(n_generations) + 1.0
            plt.plot(generation, mean_feedbacks, label=algorithm_name)
            plt.fill_between(
                generation, mean_feedbacks - std_feedbacks,
                mean_feedbacks + std_feedbacks, alpha=0.5)
            plt.yscale("symlog")
            plt.xlabel("Generation")
            plt.ylabel("Average Return")
            plt.legend()
    plt.show()


def optimize(objective_name, algorithm_name, seed):
    n_context_dims = context_dims_per_objective[objective_name]
    random_state = np.random.RandomState(seed)
    # contexts are sampled uniformly from 1 <= s <= 2 (here: < 2)
    contexts = random_state.rand(n_episodes, n_context_dims) + 1.0
    obj = objective_functions[objective_name](random_state, n_params, n_context_dims)
    initial_params = random_state.randn(n_params)
    opt = algorithms[algorithm_name](
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

    return feedbacks


if __name__ == "__main__":
    benchmark()
