import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import collections
import numpy as np
from utils import load_data

def info_datasets():
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

    file_name = "results/info_datasets.txt"
    file = open(file_name, "w")
    file.write("dataset,n_records,n_columns,n_classes,perc_majority_class,perc_minority_class\n")
    file.close()

    for dataset in datasets:
        # Load the data
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(dataset)

        X = [X_train, X_val, X_test]
        y = [y_train, y_val, y_test]
        y_flat = [item for sublist in y for item in sublist]

        n_columns = X_train.shape[1]
        n_records = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        n_classes = len(np.unique(y_flat))
        perc_min_class = min(collections.Counter(y_flat).values())*1.0/len(y_flat)
        perc_max_class = max(collections.Counter(y_flat).values())*1.0/len(y_flat)

        string = dataset + "," + str(n_records) + "," + str(n_columns) + "," + str(n_classes) + "," + \
            str(perc_max_class) + "," + str(perc_min_class) + "\n"

        file = open(file_name, "a")
        file.write(string)
        file.close()

if __name__ == "__main__":
    info_datasets()