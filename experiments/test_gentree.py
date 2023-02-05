import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pickle

from run_gentree import run_experiment
from genetic_algorithm import fitness_a

np.random.seed(42)

def run(datasets, depths, fixed_parameters, times, n_jobs, path="", save_model=False, save_path=""):
    file_name = path + "gentree.txt"

    file = open(file_name, "w")
    string = "dataset,max_depth,trees_learning_function,nbr_training_trees,vae_batch_size,vae_epochs," + \
            "fitness_function,fitness_omega,nbr_generations,ga_pop_size,mean_accuracy,std_accuracy," + \
            "mean_f1,std_f1,mean_complexity,std_complexity,best_accuracy,best_f1,best_complexity,mean_not_trees,std_not_trees," + \
            "mean_tot_time,std_tot_time,mean_vae_init_time,std_vae_init_time,mean_vae_time,std_vae_time," + \
            "mean_encoding_time,std_encoding_time,mean_gp_time,std_gp_time\n"
    file.write(string)
    file.close()

    for dataset in datasets:
        vae_pop_size = (
            fixed_parameters["nbr_training_trees"], 
            int(fixed_parameters["nbr_training_trees"]/2), 
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
            gp_population_size=fixed_parameters["ga_pop_size"]
        )

        string = dataset + "," + str(depths[dataset]) + "," + fixed_parameters["trees_learning_function"] + "," + \
            str(fixed_parameters["nbr_training_trees"]) + "," + str(fixed_parameters["vae_batch_size"]) + "," +  \
            str(fixed_parameters["vae_epochs"]) + "," + fixed_parameters["fitness_function_str"] + "," + \
            str(fixed_parameters["fitness_omega"]) + "," + str(fixed_parameters["nbr_generations"]) + "," + \
            str(fixed_parameters["ga_pop_size"]) + "," + str(result[0]) + "," + str(result[1]) + "," + \
            str(result[2]) + "," + str(result[3]) + "," + str(result[4]) + "," + str(result[5]) + "," + \
            str(result[6]) + "," + str(result[7]) + "," + str(result[8]) + "," + str(result[9]) + "," + \
            str(result[10]) + "," + str(result[11]) + "," + str(result[12]) + "," + str(result[13]) + "," + \
            str(result[14]) + "," + str(result[15]) + "," + str(result[16]) + "," + str(result[17]) + "," + \
            str(result[18]) + "," + str(result[19]) + "," + str(result[20]) + "\n"

        file = open(file_name, "a")
        file.write(string)
        file.close()

        if save_model:
            # Dump the trained decision tree classifier with Pickle
            decision_tree_pkl_filename = save_path + "{}.pkl".format(dataset)

            # Open the file to save as pkl file
            decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')

            pickle.dump(result[21], decision_tree_model_pkl)
            # Close the pickle instances
            decision_tree_model_pkl.close()

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

    run(datasets, depths, fixed_parameters, times, n_jobs, path, save_model, save_path) 