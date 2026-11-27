import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from math import inf
from timeit import default_timer as timer

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score

from gentree_torch.gentree import GenTree
from gentree_torch.utils import load_data, set_seed


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
    max_patient,
    seed=None,
    device=None,
):
    if seed is not None:
        set_seed(seed)

    start = timer()

    gt = GenTree(dataset, device=device)
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

    perc_false = (gt.false_is_tree / gt.tot_is_tree) * 100  # percentage of 'not_trees' found

    y_pred = gt.predict(X_test)
    if y_pred is not None:
        best_dts = gt.best_dt["clf"]
        complexity = gt.best_dt["n_nodes"]
        fitness_value = gt.best_dt["val_fitness_value"]
        test_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        test_f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    else:
        best_dts = 0
        complexity = 0
        fitness_value = inf
        test_accuracy = 0
        test_f1 = 0

    return [
        total_time,
        gt.vae_pop_time,
        gt.preprocessing_time,
        gt.vae_time,
        gt.encoding_time,
        gt.gp_time,
        best_dts,
        complexity,
        fitness_value,
        test_accuracy,
        test_f1,
        perc_false
    ]


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
    seed=42,
    device=None,
):
    """Runs GenTree `times` times on `dataset`.

    Returns [mean/std accuracy, mean/std f1, mean/std complexity, best accuracy, best f1,
    best complexity, mean/std not_trees, mean/std total time, mean/std vae init time,
    mean/std vae time, mean/std encoding time, mean/std gp time, best_dt,
    mean/std preprocessing time] (same layout as experiments/run_gentree.py).
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset)

    X = [X_train, X_val]
    y = [y_train, y_val]

    result_gt = Parallel(n_jobs=n_jobs)(delayed(run_gentree)(
        X, y, X_test, y_test, dataset, depth, vae_pop_size, trees_learning_function,
        epochs, fitness, omega, ngen, gp_population_size, cx_pb, mut_pb,
        min_delta, delta_patient, max_patient,
        None if seed is None else seed + i, device
    ) for i in range(0, times))

    cols = list(zip(*result_gt))
    (total_time, vae_pop_time, preprocessing_time, vae_time, encoding_time, gp_time,
     best_dts, complexity, fitness_value, test_accuracy, test_f1, not_trees) = cols

    # Best tree over the `times` runs (lowest validation fitness)
    best_dt_index = int(np.argmin(np.array(fitness_value, dtype=float)))

    def mean_std(values):
        values = np.array(values, dtype=float)
        return [np.mean(values), np.std(values)]

    return [
        *mean_std(test_accuracy),
        *mean_std(test_f1),
        *mean_std(complexity),
        test_accuracy[best_dt_index],
        test_f1[best_dt_index],
        complexity[best_dt_index],
        *mean_std(not_trees),
        *mean_std(total_time),
        *mean_std(vae_pop_time),
        *mean_std(vae_time),
        *mean_std(encoding_time),
        *mean_std(gp_time),
        best_dts[best_dt_index],
        *mean_std(preprocessing_time),
    ]
