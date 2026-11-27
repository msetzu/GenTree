import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_gentree import run_experiment
from gentree_torch.genetic_algorithm import fitness_a, fitness_b
from gentree_torch.utils import set_seed

HEADER = ("dataset,max_depth,trees_learning_function,nbr_training_trees,vae_batch_size,vae_epochs,fitness_function,"
          "fitness_omega,nbr_generations,ga_pop_size,mean_accuracy,std_accuracy,mean_f1,std_f1,mean_complexity,"
          "std_complexity,best_accuracy,best_f1,best_complexity,mean_not_trees,std_not_trees\n")


def run(
    times,
    dataset,
    depth,
    vae_pop_size,
    trees_learning_function,
    epochs,
    fitness,
    omega,
    ngen,
    gp_population_size,
    seed=42,
):
    result_gentree = run_experiment(
        n_jobs=-2,
        times=times,
        dataset=dataset,
        depth=depth,
        vae_pop_size=vae_pop_size,
        trees_learning_function=trees_learning_function,
        epochs=epochs,
        fitness=fitness,
        omega=omega,
        ngen=ngen,
        gp_population_size=gp_population_size,
        seed=seed,
    )
    # mean/std accuracy, mean/std f1, mean/std complexity, best acc/f1/complexity, mean/std not_trees
    return result_gentree[0:11]


def _sweep(file_name, param, values, datasets, depths, fitness_functions, fixed_parameters, times):
    """Runs GenTree for every (dataset, fitness, value) where `value` replaces `param`."""
    with open(file_name, "w") as file:
        file.write(HEADER)

    for dataset in datasets:
        for fitness_function in fitness_functions:
            for value in values:
                p = dict(fixed_parameters)
                p.setdefault("nbr_training_trees", 10000)
                p[param] = value
                vae_pop_size = (p["nbr_training_trees"], int(p["nbr_training_trees"] / 2), p["vae_batch_size"])

                result = run(
                    times=times,
                    dataset=dataset,
                    depth=depths[dataset],
                    vae_pop_size=vae_pop_size,
                    trees_learning_function=p["trees_learning_function"],
                    epochs=p["vae_epochs"],
                    fitness=fitness_functions[fitness_function],
                    omega=p["fitness_omega"],
                    ngen=p["nbr_generations"],
                    gp_population_size=p["ga_pop_size"]
                )

                row = [dataset, depths[dataset], p["trees_learning_function"], p["nbr_training_trees"],
                       p["vae_batch_size"], p["vae_epochs"], fitness_function, p["fitness_omega"],
                       p["nbr_generations"], p["ga_pop_size"]] + list(result)

                with open(file_name, "a") as file:
                    file.write(",".join(str(v) for v in row) + "\n")


def nbr_training_trees(datasets, depths, fitness_functions, fixed_parameters, times, path=""):
    _sweep(path + "nbr_training_trees.txt", "nbr_training_trees", [100, 1000, 10000, 50000],
           datasets, depths, fitness_functions, fixed_parameters, times)


def trees_learning_functions(datasets, depths, fitness_functions, fixed_parameters, times, path=""):
    _sweep(path + "trees_learning_functions.txt", "trees_learning_function", ["RF", "DT"],
           datasets, depths, fitness_functions, fixed_parameters, times)


def ga_pop_size(datasets, depths, fitness_functions, fixed_parameters, times, path=""):
    _sweep(path + "ga_pop_size.txt", "ga_pop_size", [50, 100, 250, 500],
           datasets, depths, fitness_functions, fixed_parameters, times)


def omega(datasets, depths, fitness_functions, fixed_parameters, times, path=""):
    _sweep(path + "omega.txt", "fitness_omega", [0.0, 0.25, 0.5, 0.75, 1.0],
           datasets, depths, fitness_functions, fixed_parameters, times)


def nbr_generations(datasets, depths, fitness_functions, fixed_parameters, times, path=""):
    _sweep(path + "nbr_generations.txt", "nbr_generations", [1, 10, 25, 50, 100],
           datasets, depths, fitness_functions, fixed_parameters, times)


if __name__ == "__main__":

    datasets = [
        "pima",
        "car",
        "drybean",
        "iris"
    ]

    depths = {
        "pima": 5,
        "car": 8,
        "drybean": 7,
        "iris": 4
    }

    fixed_parameters = {
        "trees_learning_function": "DT",
        "vae_batch_size": 32,
        "vae_epochs": 100,
        "fitness_omega": 0.75,
        "nbr_generations": 50,
        "ga_pop_size": 100
    }

    path = "results/hyperparameters_tuning/"
    os.makedirs(path, exist_ok=True)
    fitness_functions = {"fitness_a": fitness_a, "fitness_b": fitness_b}
    times = 10

    set_seed(42)
    nbr_training_trees(datasets, depths, fitness_functions, fixed_parameters, times, path)
    trees_learning_functions(datasets, depths, fitness_functions, fixed_parameters, times, path)
    ga_pop_size(datasets, depths, fitness_functions, fixed_parameters, times, path)
    omega(datasets, depths, fitness_functions, fixed_parameters, times, path)
    nbr_generations(datasets, depths, fitness_functions, fixed_parameters, times, path)
