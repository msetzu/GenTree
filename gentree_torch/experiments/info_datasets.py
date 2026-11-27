import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import collections

import numpy as np

from gentree_torch.utils import load_data


def info_datasets(path="results/"):
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

    os.makedirs(path, exist_ok=True)
    file_name = path + "info_datasets.txt"
    with open(file_name, "w") as file:
        file.write("dataset,n_records,n_columns,n_classes,perc_majority_class,perc_minority_class\n")

    for dataset in datasets:
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset)

        y_flat = np.concatenate([y_train, y_val, y_test]).tolist()
        counts = collections.Counter(y_flat).values()

        n_columns = X_train.shape[1]
        n_records = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        n_classes = len(np.unique(y_flat))
        perc_min_class = min(counts) * 1.0 / len(y_flat)
        perc_max_class = max(counts) * 1.0 / len(y_flat)

        with open(file_name, "a") as file:
            file.write(",".join(str(v) for v in
                                [dataset, n_records, n_columns, n_classes, perc_max_class, perc_min_class]) + "\n")


if __name__ == "__main__":
    info_datasets()
