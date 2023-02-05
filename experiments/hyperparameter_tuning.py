import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np

from utils import load_data
from genetic_algorithm import fitness_a, fitness_b
from run_gentree import run_experiment

np.random.seed(42)

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
    gp_population_size
):

    # Load the data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset)

    X = [X_train, X_val]
    y = [y_train, y_val]

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
                                gp_population_size=gp_population_size
                            )

    result = [
        result_gentree[0],  # mean_accuracy
        result_gentree[1],  # std_accuracy
        result_gentree[2],  # mean_f1
        result_gentree[3],  # std_f1
        result_gentree[4],  # mean_complexity
        result_gentree[5],  # std_complexity
        result_gentree[6],  # best_accuracy
        result_gentree[7],  # best_f1
        result_gentree[8],  # best_complexity
        result_gentree[9],  # mean_not_trees
        result_gentree[10]  # std_not_trees
    ]

    return result

def nbr_training_trees(datasets, depths, fitness_functions, fixed_parameters, times):
    file_name = path + "nbr_training_trees.txt"

    nbr_training_trees_values = [100, 1000, 10000, 50000]

    file = open(file_name, "w")
    file.write("dataset,max_depth,trees_learning_function,nbr_training_trees,vae_batch_size,vae_epochs,fitness_function,fitness_omega,nbr_generations,ga_pop_size,mean_accuracy,std_accuracy,mean_f1,std_f1,mean_complexity,std_complexity,best_accuracy,best_f1,best_complexity,mean_not_trees,std_not_trees\n")
    file.close()

    for dataset in datasets:
        for fitness_function in fitness_functions:
            for nbr_training_trees_value in nbr_training_trees_values:
                vae_pop_size = (nbr_training_trees_value, int(nbr_training_trees_value/2), fixed_parameters["vae_batch_size"])
                result = run(
                    times=times, 
                    dataset=dataset, 
                    depth=depths[dataset], 
                    vae_pop_size=vae_pop_size, 
                    trees_learning_function=fixed_parameters["trees_learning_function"],
                    epochs=fixed_parameters["vae_epochs"], 
                    fitness=fitness_functions[fitness_function], 
                    omega=fixed_parameters["fitness_omega"], 
                    ngen=fixed_parameters["nbr_generations"], 
                    gp_population_size=fixed_parameters["ga_pop_size"]
                )

                string = dataset + "," + str(depths[dataset]) + "," + fixed_parameters["trees_learning_function"] + "," + \
                        str(nbr_training_trees_value) + "," + str(fixed_parameters["vae_batch_size"]) + "," + \
                        str(fixed_parameters["vae_epochs"]) + "," + fitness_function + "," + \
                        str(fixed_parameters["fitness_omega"]) + "," + str(fixed_parameters["nbr_generations"]) + "," + \
                        str(fixed_parameters["ga_pop_size"]) + "," + str(result[0]) + "," + str(result[1]) + "," + \
                        str(result[2]) + "," + str(result[3]) + "," + str(result[4]) + "," + str(result[5]) + "," + \
                        str(result[6]) + "," + str(result[7]) + "," + str(result[8]) + "," + str(result[9]) + "," + \
                        str(result[10]) + "\n"

                file = open(file_name, "a")
                file.write(string)
                file.close()

def trees_learning_functions(datasets, depths, fitness_functions, fixed_parameters, times, path=""):
    file_name = path + "trees_learning_functions.txt"

    trees_learning_functions_values = ["RF", "DT"]
    nbr_training_trees = 10000

    file = open(file_name, "w")
    file.write("dataset,max_depth,trees_learning_function,nbr_training_trees,vae_batch_size,vae_epochs,fitness_function,fitness_omega,nbr_generations,ga_pop_size,mean_accuracy,std_accuracy,mean_f1,std_f1,mean_complexity,std_complexity,best_accuracy,best_f1,best_complexity,mean_not_trees,std_not_trees\n")
    file.close()

    for dataset in datasets:
        for fitness_function in fitness_functions:
            for trees_learning_functions_value in trees_learning_functions_values:
                vae_pop_size = (nbr_training_trees, int(nbr_training_trees/2), fixed_parameters["vae_batch_size"])
                result = run(
                    times=times, 
                    dataset=dataset, 
                    depth=depths[dataset], 
                    vae_pop_size=vae_pop_size, 
                    trees_learning_function=trees_learning_functions_value,
                    epochs=fixed_parameters["vae_epochs"], 
                    fitness=fitness_functions[fitness_function], 
                    omega=fixed_parameters["fitness_omega"], 
                    ngen=fixed_parameters["nbr_generations"], 
                    gp_population_size=fixed_parameters["ga_pop_size"]
                )

                string = dataset + "," + str(depths[dataset]) + "," + trees_learning_functions_value + "," + \
                        str(nbr_training_trees) + "," + str(fixed_parameters["vae_batch_size"]) + "," + \
                        str(fixed_parameters["vae_epochs"]) + "," + fitness_function + "," + \
                        str(fixed_parameters["fitness_omega"]) + "," + str(fixed_parameters["nbr_generations"]) + "," + \
                        str(fixed_parameters["ga_pop_size"]) + "," + str(result[0]) + "," + str(result[1]) + "," + \
                        str(result[2]) + "," + str(result[3]) + "," + str(result[4]) + "," + str(result[5]) + "," + \
                        str(result[6]) + "," + str(result[7]) + "," + str(result[8]) + "," + str(result[9]) + "," + \
                        str(result[10]) + "\n"

                file = open(file_name, "a")
                file.write(string)
                file.close()

def ga_pop_size(datasets, depths, fitness_functions, fixed_parameters, times, path=""):
    file_name = path + "ga_pop_size.txt"

    ga_pop_size_values = [50, 100, 250, 500]
    nbr_training_trees = 10000

    file = open(file_name, "w")
    file.write("dataset,max_depth,trees_learning_function,nbr_training_trees,vae_batch_size,vae_epochs,fitness_function,fitness_omega,nbr_generations,ga_pop_size,mean_accuracy,std_accuracy,mean_f1,std_f1,mean_complexity,std_complexity,best_accuracy,best_f1,best_complexity,mean_not_trees,std_not_trees\n")
    file.close()

    for dataset in datasets:
        for fitness_function in fitness_functions:
            for ga_pop_size_value in ga_pop_size_values:
                vae_pop_size = (nbr_training_trees, int(nbr_training_trees/2), fixed_parameters["vae_batch_size"])
                result = run(
                    times, 
                    dataset, 
                    depths[dataset], 
                    vae_pop_size, 
                    fixed_parameters["trees_learning_function"],
                    fixed_parameters["vae_epochs"], 
                    fitness_functions[fitness_function], 
                    fixed_parameters["fitness_omega"], 
                    fixed_parameters["nbr_generations"], 
                    ga_pop_size_value
                )

                string = dataset + "," + str(depths[dataset]) + "," + fixed_parameters["trees_learning_function"] + "," + \
                        str(nbr_training_trees) + "," + str(fixed_parameters["vae_batch_size"]) + "," + \
                        str(fixed_parameters["vae_epochs"]) + "," + fitness_function + "," + \
                        str(fixed_parameters["fitness_omega"]) + "," + str(fixed_parameters["nbr_generations"]) + "," + \
                        str(ga_pop_size_value) + "," + str(result[0]) + "," + str(result[1]) + "," + \
                        str(result[2]) + "," + str(result[3]) + "," + str(result[4]) + "," + str(result[5]) + "," + \
                        str(result[6]) + "," + str(result[7]) + "," + str(result[8]) + "," + str(result[9]) + "," + \
                        str(result[10]) + "\n"

                file = open(file_name, "a")
                file.write(string)
                file.close()

def omega(datasets, depths, fitness_functions, fixed_parameters, times, path=""):
    file_name = path + "omega.txt"

    omega_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    nbr_training_trees = 10000

    file = open(file_name, "w")
    file.write("dataset,max_depth,trees_learning_function,nbr_training_trees,vae_batch_size,vae_epochs,fitness_function,fitness_omega,nbr_generations,ga_pop_size,mean_accuracy,std_accuracy,mean_f1,std_f1,mean_complexity,std_complexity,best_accuracy,best_f1,best_complexity,mean_not_trees,std_not_trees\n")
    file.close()

    for dataset in datasets:
        for fitness_function in fitness_functions:
            for omega_value in omega_values:
                vae_pop_size = (nbr_training_trees, int(nbr_training_trees/2), fixed_parameters["vae_batch_size"])
                result = run(
                    times, 
                    dataset, 
                    depths[dataset], 
                    vae_pop_size, 
                    fixed_parameters["trees_learning_function"],
                    fixed_parameters["vae_epochs"], 
                    fitness_functions[fitness_function], 
                    omega_value, 
                    fixed_parameters["nbr_generations"], 
                    fixed_parameters["ga_pop_size"]
                )

                string = dataset + "," + str(depths[dataset]) + "," + fixed_parameters["trees_learning_function"] + "," + \
                        str(nbr_training_trees) + "," + str(fixed_parameters["vae_batch_size"]) + "," + \
                        str(fixed_parameters["vae_epochs"]) + "," + fitness_function + "," + \
                        str(omega_value) + "," + str(fixed_parameters["nbr_generations"]) + "," + \
                        str(fixed_parameters["ga_pop_size"]) + "," + str(result[0]) + "," + str(result[1]) + "," + \
                        str(result[2]) + "," + str(result[3]) + "," + str(result[4]) + "," + str(result[5]) + "," + \
                        str(result[6]) + "," + str(result[7]) + "," + str(result[8]) + "," + str(result[9]) + "," + \
                        str(result[10]) + "\n"

                file = open(file_name, "a")
                file.write(string)
                file.close()

def nbr_generations(datasets, depths, fitness_functions, fixed_parameters, times, path=""):
    file_name = path + "nbr_generations.txt"

    nbr_generations_values = [1, 10, 25, 50, 100]
    nbr_training_trees = 10000

    file = open(file_name, "w")
    file.write("dataset,max_depth,trees_learning_function,nbr_training_trees,vae_batch_size,vae_epochs,fitness_function,fitness_omega,nbr_generations,ga_pop_size,mean_accuracy,std_accuracy,mean_f1,std_f1,mean_complexity,std_complexity,best_accuracy,best_f1,best_complexity,mean_not_trees,std_not_trees\n")
    file.close()

    for dataset in datasets:
        for fitness_function in fitness_functions:
            for nbr_generations_value in nbr_generations_values:
                vae_pop_size = (nbr_training_trees, int(nbr_training_trees/2), fixed_parameters["vae_batch_size"])
                result = run(
                    times, 
                    dataset, 
                    depths[dataset], 
                    vae_pop_size, 
                    fixed_parameters["trees_learning_function"],
                    fixed_parameters["vae_epochs"], 
                    fitness_functions[fitness_function], 
                    fixed_parameters["fitness_omega"], 
                    nbr_generations_value, 
                    fixed_parameters["ga_pop_size"]
                )

                string = dataset + "," + str(depths[dataset]) + "," + fixed_parameters["trees_learning_function"] + ","  + \
                        str(nbr_training_trees) + "," + str(fixed_parameters["vae_batch_size"]) + "," + \
                        str(fixed_parameters["vae_epochs"]) + "," + fitness_function + "," + \
                        str(fixed_parameters["fitness_omega"]) + "," + str(nbr_generations_value) + "," + \
                        str(fixed_parameters["ga_pop_size"]) + "," + str(result[0]) + "," + str(result[1]) + "," + \
                        str(result[2]) + "," + str(result[3]) + "," + str(result[4]) + "," + str(result[5]) + "," + \
                        str(result[6]) + "," + str(result[7]) + "," + str(result[8]) + "," + str(result[9]) + "," + \
                        str(result[10]) + "\n"

                file = open(file_name, "a")
                file.write(string)
                file.close()
    
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
    fitness_functions = {"fitness_a": fitness_a, "fitness_b": fitness_b}
    times = 10

    nbr_training_trees(datasets, depths, fitness_functions, fixed_parameters, times, path) 
    trees_learning_functions(datasets, depths, fitness_functions, fixed_parameters, times, path) 
    ga_pop_size(datasets, depths, fitness_functions, fixed_parameters, times, path) 
    omega(datasets, depths, fitness_functions, fixed_parameters, times, path) 
    nbr_generations(datasets, depths, fitness_functions, fixed_parameters, times, path) 