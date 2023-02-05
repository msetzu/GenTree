import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from joblib import Parallel, delayed
from math import inf
from timeit import default_timer as timer

from gentree import GenTree
from utils import load_data
from genetic_algorithm import fitness_a, fitness_b

np.random.seed(42)

def run_gentree(
    X, 
    y, 
    X_test, 
    y_test, 
    dataset, 
    depth, 
    vae_pop_size, 
    trees_learning_function,
    epochs, 
    fitness, 
    omega, 
    ngen, 
    gp_population_size,
    cx_pb,
    mut_pb,
    min_delta,
    delta_patient,
    max_patient
):
    start = timer()

    gt = GenTree(dataset)
    gt.run(
        X=X, 
        y=y, 
        depth=depth, 
        fitness=fitness,
        vae_pop_size=vae_pop_size,
        trees_learning_function=trees_learning_function, 
        epochs=epochs,  
        min_delta=min_delta,
        delta_patient=delta_patient,
        max_patient=max_patient,
        omega=omega, 
        ngen=ngen, 
        population_size=gp_population_size,
        cx_pb=cx_pb,
        mut_pb=mut_pb
    )

    stop = timer()
    total_time = (stop - start)

    n_is_tree = gt.tot_is_tree
    n_false = gt.false_is_tree
    perc_false = (n_false/n_is_tree)*100 # percentage of 'not_trees' found

    # Best dt
    y_pred = gt.predict(X_test) 
    if y_pred is not None:
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')

        vae_pop_time = gt.vae_pop_time
        vae_time = gt.vae_time
        preprocessing_time = gt.preprocessing_time
        encoding_time = gt.encoding_time
        gp_time = gt.gp_time
        best_dts = gt.best_dt["clf"]
        complexity = gt.best_dt["n_nodes"]
        fitness_value = gt.best_dt["n_nodes"]
        test_accuracy = accuracy
        test_f1 = f1
    else:
        accuracy = 0
        f1 = 0

        vae_pop_time = gt.vae_pop_time
        vae_time = gt.vae_time
        preprocessing_time = gt.preprocessing_time
        encoding_time = gt.encoding_time
        gp_time = gt.gp_time
        best_dts = 0
        complexity = 0
        fitness_value = inf
        test_accuracy = 0
        test_f1 = 0

    result = [
        total_time,
        vae_pop_time, 
        preprocessing_time,
        vae_time, 
        encoding_time,
        gp_time, 
        best_dts, 
        complexity, 
        fitness_value, 
        test_accuracy, 
        test_f1, 
        perc_false
    ]

    return result

def run_experiment(
    n_jobs,
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
    cx_pb=0.6,
    mut_pb=0.001,
    min_delta=0.1, 
    delta_patient=2,
    max_patient=2,
):

    # Load the data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset)

    X = [X_train, X_val]
    y = [y_train, y_val]

    result_gt = Parallel(n_jobs=n_jobs)(delayed(run_gentree)(
                                                    X, 
                                                    y, 
                                                    X_test, 
                                                    y_test, 
                                                    dataset, 
                                                    depth, 
                                                    vae_pop_size, 
                                                    trees_learning_function,
                                                    epochs, 
                                                    fitness, 
                                                    omega, 
                                                    ngen, 
                                                    gp_population_size,
                                                    cx_pb,
                                                    mut_pb,
                                                    min_delta,
                                                    delta_patient,
                                                    max_patient
                                                    ) for i in range(0, times))

    total_time = [item[0] for item in result_gt]
    vae_pop_time = [item[1] for item in result_gt]
    preprocessing_time = [item[2] for item in result_gt]
    vae_time = [item[3] for item in result_gt]
    encoding_time = [item[4] for item in result_gt]
    gp_time = [item[5] for item in result_gt]
    best_dts = [item[6] for item in result_gt]
    complexity = [item[7] for item in result_gt]
    fitness_value = [item[8] for item in result_gt]
    test_accuracy = [item[9] for item in result_gt]
    test_f1 = [item[10] for item in result_gt]
    not_trees = [item[11] for item in result_gt]

    # Find the best tree on 'times' times
    best_dt_index = int(np.argmin(np.array(fitness_value)))
    best_dt = best_dts[best_dt_index]

    mean_accuracy = np.mean(np.array(test_accuracy))
    std_accuracy = np.std(np.array(test_accuracy))

    mean_f1 = np.mean(np.array(test_f1))
    std_f1 = np.std(np.array(test_f1))

    mean_complexity = np.mean(np.array(complexity))
    std_complexity = np.std(np.array(complexity))

    mean_total_time = np.mean(np.array(total_time))
    std_total_time = np.std(np.array(total_time))

    mean_vae_pop_time = np.mean(np.array(vae_pop_time))
    std_vae_pop_time = np.std(np.array(vae_pop_time))

    mean_preprocessing_time = np.mean(np.array(preprocessing_time))
    std_preprocessing_time = np.std(np.array(preprocessing_time))

    mean_vae_time = np.mean(np.array(vae_time))
    std_vae_time = np.std(np.array(vae_time))

    mean_encoding_time = np.mean(np.array(encoding_time))
    std_encoding_time = np.std(np.array(encoding_time))

    mean_gp_time = np.mean(np.array(gp_time))
    std_gp_time = np.std(np.array(gp_time))

    best_accuracy = test_accuracy[best_dt_index]
    best_f1 = test_f1[best_dt_index]
    best_complexity = complexity[best_dt_index]

    mean_not_trees = np.mean(np.array(not_trees))
    std_not_trees = np.std(np.array(not_trees))

    result = [
        mean_accuracy, 
        std_accuracy, 
        mean_f1, 
        std_f1, 
        mean_complexity, 
        std_complexity, 
        best_accuracy, 
        best_f1, 
        best_complexity,
        mean_not_trees,
        std_not_trees,
        mean_total_time,
        std_total_time,
        mean_vae_pop_time,
        std_vae_pop_time,
        mean_vae_time,
        std_vae_time,
        mean_encoding_time,
        std_encoding_time,
        mean_gp_time,
        std_gp_time,
        best_dt,
        mean_preprocessing_time,
        std_preprocessing_time
    ]

    return result