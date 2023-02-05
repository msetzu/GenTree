import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from timeit import default_timer as timer

from tree_encoder import JakowskiEncoder
from vae import CVAE
from genetic_algorithm import GeneticProcess
from vae_dataset_generator import initial_population

class GenTree:
    def __init__(self, dataset):
        self.dataset = dataset
        self.is_fitted = False

    def run(
        self, 
        X, 
        y, 
        depth, 
        fitness, 
        vae_pop_size, 
        trees_learning_function="DT", 
        epochs=100,    
        min_delta=0.1, #0.05
        delta_patient=3,
        max_patient=2,    
        omega=0.75, 
        ngen=50, 
        population_size=100, 
        cx_pb=0.6, 
        mut_pb=0.001, 
        same_depth=False,
        initial_tree=None
    ):

        X_train = X[0]
        X_val = X[1]

        y_train = y[0]
        y_val = y[1]

        n_features_in = X_train.shape[1]
        n_classes = len(np.unique(y_train))

        train_size = vae_pop_size[0]
        test_size = vae_pop_size[1]
        batch_size = vae_pop_size[2]

        self.encoder = JakowskiEncoder(n_features=n_features_in, n_classes=n_classes)

        start = timer()
        # vae_train, vae_test, n_nodes, n_leaves, depths, encoding_time
        result_init = initial_population(
                                        X=X_train, 
                                        y=y_train, 
                                        max_depth=depth,
                                        train_size=train_size, 
                                        test_size=test_size, 
                                        encoder=self.encoder,
                                        mode = trees_learning_function,
                                        same_depth=same_depth
                                    )

        stop = timer()
        self.vae_pop_time = (stop - start)                      # Time to generate the VAE training set

        self.encoding_time = result_init[5]                     # Time to encode all trees in the VAE training set.
        max_depth = max(result_init[4])                         # Maximum depth of the training trees

        start = timer()

        vae_train = np.transpose(result_init[0], (0, 2, 1))
        vae_test = np.transpose(result_init[1], (0, 2, 1))

        # Scaling 
        self.scalers = {}
        for i in range(vae_train.shape[2]):
            self.scalers[i] = MinMaxScaler()
            vae_train[:, :, i] = self.scalers[i].fit_transform(vae_train[:, :, i]) 

        for i in range(vae_test.shape[2]):
            vae_test[:, :, i] = self.scalers[i].transform(vae_test[:, :, i]) 
            
        vae_train = vae_train.astype('float32')
        vae_test = vae_test.astype('float32')

        train_dataset = (tf.data.Dataset.from_tensor_slices(vae_train).shuffle(train_size).batch(batch_size))
        test_dataset = (tf.data.Dataset.from_tensor_slices(vae_test).shuffle(test_size).batch(batch_size))

        stop = timer()
        self.preprocessing_time = (stop - start)

        start = timer()

        # VAE model and training
        self.vae = CVAE(input_shape=(vae_train.shape[1], vae_train.shape[2]))
        self.vae.fit(
                epochs=epochs, 
                train_dataset=train_dataset, 
                test_dataset=test_dataset, 
                min_delta=min_delta,
                delta_patient=delta_patient,
                max_patient=max_patient
        )

        stop = timer()
        self.vae_time = (stop - start) 

        start = timer()
        gp = GeneticProcess(
            model=self.vae, 
            scalers=self.scalers, 
            encoder=self.encoder, 
            max_depth=max_depth, 
            X_test=X_val, 
            y_test=y_val,
            fitness=fitness,
            initial_tree=initial_tree,
            mutation="custom",
            omega=omega,
            ngen=ngen,
            population_size=population_size,
            cx_pb=cx_pb, 
            mut_pb=mut_pb, 
            plot=False
        )

        gp.run()

        stop = timer()
        self.gp_time = (stop - start)

        self.hof = gp.hof()

        if self.hof is not None: 
            self.is_fitted = True
            # Best individual found
            self.best_dt = {
                        "clf": self.hof[0][0],
                        "n_nodes": self.hof[0][1],
                        "n_leaves": self.hof[0][2],
                        "max_depth": self.hof[0][3],
                        "val_errors": self.hof[0][4],
                        "val_accuracy": self.hof[0][5],
                        "val_f1": self.hof[0][6],
                        "val_fitness_value": self.hof[0][7]
                    }
        else: # If GP didn't even find a real tree
            self.best_dt = None

        self.tot_is_tree = gp.tot_is_tree     # Number of times "isTree" has been called
        self.false_is_tree = gp.false_is_tree # Number of times "isTree" returns False

    def predict(self, X):
        if self.is_fitted:
            clf = self.best_dt["clf"]
            return clf.predict(X)
        else:
            return None

        
