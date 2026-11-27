import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle

from run_gentree import run_experiment
from gentree_torch.genetic_algorithm import fitness_a
from gentree_torch.utils import set_seed

HEADER = ("dataset,max_depth,trees_learning_function,nbr_training_trees,vae_batch_size,vae_epochs,"
          "fitness_function,fitness_omega,nbr_generations,ga_pop_size,mean_accuracy,std_accuracy,"
          "mean_f1,std_f1,mean_complexity,std_complexity,best_accuracy,best_f1,best_complexity,mean_not_trees,std_not_trees,"
          "mean_tot_time,std_tot_time,mean_vae_init_time,std_vae_init_time,mean_vae_time,std_vae_time,"
          "mean_encoding_time,std_encoding_time,mean_gp_time,std_gp_time\n")


def run(datasets, depths, fixed_parameters, times, n_jobs, path="", save_model=False, save_path="", seed=42):
    os.makedirs(path or ".", exist_ok=True)
    if save_model:
        os.makedirs(save_path or ".", exist_ok=True)

    file_name = path + "gentree.txt"
    with open(file_name, "w") as file:
        file.write(HEADER)

    for dataset in datasets:
        vae_pop_size = (
            fixed_parameters["nbr_training_trees"],
            int(fixed_parameters["nbr_training_trees"] / 2),
            fixed_parameters["vae_batch_size"]
        )

        result = run_experiment(
            n_jobs=n_jobs,
            times=times,
            dataset=dataset,
            depth=depths[dataset],
            vae_pop_size=vae_pop_size,
            trees_learning_function=fixed_parameters["trees_learning_function"],
            epochs=fixed_parameters["vae_epochs"],
            fitness=fixed_parameters["fitness_function"],
            omega=fixed_parameters["fitness_omega"],
            ngen=fixed_parameters["nbr_generations"],
            gp_population_size=fixed_parameters["ga_pop_size"],
            seed=seed,
        )

        row = [dataset, depths[dataset], fixed_parameters["trees_learning_function"],
               fixed_parameters["nbr_training_trees"], fixed_parameters["vae_batch_size"],
               fixed_parameters["vae_epochs"], fixed_parameters["fitness_function_str"],
               fixed_parameters["fitness_omega"], fixed_parameters["nbr_generations"],
               fixed_parameters["ga_pop_size"]] + result[0:21]

        with open(file_name, "a") as file:
            file.write(",".join(str(v) for v in row) + "\n")

        if save_model:
            # Dump the best decision tree classifier with Pickle
            with open(save_path + "{}.pkl".format(dataset), 'wb') as f:
                pickle.dump(result[21], f)


if __name__ == "__main__":

    datasets = [
        "iris",
        "led7",
        "car",
        "pima",
        "australian",
        "lymph",
        "breast",
        "heart",
        "ecoli",
        "bank",
        "sonar",
        "avila",
        "banknote",
        "drybean",
        "isolet",
        "wine",
        "yeast",
        "glass",
        "vehicle",
        "egg"
    ]

    depths = {
        "iris": 4,
        "led7": 7,
        "car": 8,
        "pima": 4,
        "australian": 4,
        "lymph": 4,
        "breast": 6,
        "heart": 4,
        "ecoli": 4,
        "bank": 4,
        "sonar": 7,
        "avila": 15,
        "banknote": 6,
        "drybean": 7,
        "isolet": 10,
        "wine": 9,
        "yeast": 6,
        "glass": 4,
        "vehicle": 10,
        "egg": 12
    }

    fixed_parameters = {
        "trees_learning_function": "DT",
        "nbr_training_trees": 10000,
        "vae_batch_size": 32,
        "vae_epochs": 100,
        "fitness_function": fitness_a,
        "fitness_function_str": "fitness_a",
        "fitness_omega": 0.75,
        "nbr_generations": 10,
        "ga_pop_size": 100
    }

    times = 10
    n_jobs = 1
    save_model = True
    save_path = "results/confronti/gentree/"
    path = "results/confronti/"

    set_seed(42)
    run(datasets, depths, fixed_parameters, times, n_jobs, path, save_model, save_path)
