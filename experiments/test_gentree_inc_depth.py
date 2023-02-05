import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pickle

from run_gentree import run_experiment
from genetic_algorithm import fitness_a, fitness_b

np.random.seed(42)

def run(datasets, depths, fixed_parameters, fitness_functions, times, n_jobs, path, save_model=False, save_path=""):
    file_name = path + "gentree_inc_depth.txt"

    file = open(file_name, "w")
    string = "dataset,max_depth,trees_learning_function,nbr_training_trees,vae_batch_size,vae_epochs," + \
            "fitness_function,fitness_omega,nbr_generations,ga_pop_size,mean_accuracy,std_accuracy," + \
            "mean_f1,std_f1,mean_complexity,std_complexity,best_accuracy,best_f1,best_complexity,mean_not_trees,std_not_trees," + \
            "mean_vae_init_time,std_vae_init_time,mean_vae_time,std_vae_time,mean_encoding_time,std_encoding_time," + \
            "mean_gp_time,std_gp_time\n"
    file.write(string)
    file.close()

    for fitness_function in fitness_functions:
        for dataset in datasets:
            vae_pop_size = (
                fixed_parameters["nbr_training_trees"], 
                int(fixed_parameters["nbr_training_trees"]/2), 
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
                gp_population_size=fixed_parameters["ga_pop_size"]
            )

            string = dataset + "," + str(depth) + "," + fixed_parameters["trees_learning_function"] + "," + \
                str(fixed_parameters["nbr_training_trees"]) + "," + str(fixed_parameters["vae_batch_size"]) + "," +  \
                str(fixed_parameters["vae_epochs"]) + "," + fitness_function + "," + \
                str(fixed_parameters["fitness_omega"]) + "," + str(fixed_parameters["nbr_generations"]) + "," + \
                str(fixed_parameters["ga_pop_size"]) + "," + str(result[0]) + "," + str(result[1]) + "," + \
                str(result[2]) + "," + str(result[3]) + "," + str(result[4]) + "," + str(result[5]) + "," + \
                str(result[6]) + "," + str(result[7]) + "," + str(result[8]) + "," + str(result[9]) + "," + \
                str(result[10]) + "," + str(result[11]) + "," + str(result[12]) + "," + str(result[13]) + "," + \
                str(result[14]) + "," + str(result[15]) + "," + str(result[16]) + "," + str(result[17]) + "," + \
                str(result[18]) + "\n"

            file = open(file_name, "a")
            file.write(string)
            file.close()

            if save_model:
                # Dump the trained decision tree classifier with Pickle
                decision_tree_pkl_filename = save_path + "{}_{}.pkl".format(dataset, fitness_function)

                # Open the file to save as pkl file
                decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')

                pickle.dump(result[19], decision_tree_model_pkl)
                # Close the pickle instances
                decision_tree_model_pkl.close()

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

    run(datasets, depths, fixed_parameters, fitness_functions, times, n_jobs, path, save_model, save_path) 