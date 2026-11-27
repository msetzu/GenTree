import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle

from run_gentree import run_experiment
from test_gentree import HEADER
from gentree_torch.genetic_algorithm import fitness_a, fitness_b
from gentree_torch.utils import set_seed


def run(datasets, depths, fixed_parameters, fitness_functions, times, n_jobs, path, save_model=False, save_path="",
        seed=42):
    os.makedirs(path or ".", exist_ok=True)
    if save_model:
        os.makedirs(save_path or ".", exist_ok=True)

    file_name = path + "gentree_inc_depth.txt"
    with open(file_name, "w") as file:
        file.write(HEADER)

    for fitness_function in fitness_functions:
        for dataset in datasets:
            vae_pop_size = (
                fixed_parameters["nbr_training_trees"],
                int(fixed_parameters["nbr_training_trees"] / 2),
                fixed_parameters["vae_batch_size"]
            )

            depth = depths[dataset] + 2

            result = run_experiment(
                n_jobs=n_jobs,
                times=times,
                dataset=dataset,
                depth=depth,
                vae_pop_size=vae_pop_size,
                trees_learning_function=fixed_parameters["trees_learning_function"],
                epochs=fixed_parameters["vae_epochs"],
                fitness=fitness_functions[fitness_function],
                omega=fixed_parameters["fitness_omega"],
                ngen=fixed_parameters["nbr_generations"],
                gp_population_size=fixed_parameters["ga_pop_size"],
                seed=seed,
            )

            row = [dataset, depth, fixed_parameters["trees_learning_function"],
                   fixed_parameters["nbr_training_trees"], fixed_parameters["vae_batch_size"],
                   fixed_parameters["vae_epochs"], fitness_function,
                   fixed_parameters["fitness_omega"], fixed_parameters["nbr_generations"],
                   fixed_parameters["ga_pop_size"]] + result[0:21]

            with open(file_name, "a") as file:
                file.write(",".join(str(v) for v in row) + "\n")

            if save_model:
                with open(save_path + "{}_{}.pkl".format(dataset, fitness_function), 'wb') as f:
                    pickle.dump(result[21], f)  # best decision tree


if __name__ == "__main__":

    datasets = [
        "pima",
        "iris",
        "car",
        "drybean"
    ]

    depths = {
        "pima": 5,
        "car": 8,
        "drybean": 7,
        "iris": 4
    }

    fixed_parameters = {
        "trees_learning_function": "NEW_RF",
        "nbr_training_trees": 10000,
        "vae_batch_size": 32,
        "vae_epochs": 100,
        "fitness_omega": 0.75,
        "nbr_generations": 10,
        "ga_pop_size": 100
    }

    times = 10
    n_jobs = 1
    save_model = False
    fitness_functions = {"fitness_a": fitness_a, "fitness_b": fitness_b}
    save_path = "results/confronti/gentree_inc_depth/"
    path = "results/confronti/"

    set_seed(42)
    run(datasets, depths, fixed_parameters, fitness_functions, times, n_jobs, path, save_model, save_path)
