from timeit import default_timer as timer

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .genetic_algorithm import GeneticProcess
from .tree_encoder import JakowskiEncoder
from .vae import CVAE, make_loader
from .vae_dataset_generator import initial_population


class GenTree:
    def __init__(self, dataset, device=None):
        self.dataset = dataset
        self.device = device  # None -> "cuda" if available, else "cpu"
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
        min_delta=0.1,
        delta_patient=3,
        max_patient=2,
        omega=0.75,
        ngen=50,
        population_size=100,
        cx_pb=0.6,
        mut_pb=0.001,
        same_depth=False,
        initial_tree=None,
        verbose=True,
    ):
        X_train, X_val = X[0], X[1]
        y_train, y_val = y[0], y[1]

        n_features_in = X_train.shape[1]
        n_classes = len(np.unique(y_train))

        train_size, test_size, batch_size = vae_pop_size

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
            mode=trees_learning_function,
            same_depth=same_depth
        )
        stop = timer()
        self.vae_pop_time = (stop - start)          # Time to generate the VAE training set

        self.encoding_time = result_init[5]         # Time to encode all trees in the VAE training set
        max_depth = max(result_init[4])             # Maximum depth of the training trees

        start = timer()

        vae_train = np.transpose(result_init[0], (0, 2, 1))  # (n, length, 2)
        vae_test = np.transpose(result_init[1], (0, 2, 1))

        # Scaling (one MinMaxScaler per channel, fitted per position)
        self.scalers = {}
        for i in range(vae_train.shape[2]):
            self.scalers[i] = MinMaxScaler()
            vae_train[:, :, i] = self.scalers[i].fit_transform(vae_train[:, :, i])

        for i in range(vae_test.shape[2]):
            vae_test[:, :, i] = self.scalers[i].transform(vae_test[:, :, i])

        vae_train = vae_train.astype('float32')
        vae_test = vae_test.astype('float32')

        train_dataset = make_loader(vae_train, batch_size=batch_size, shuffle=True)
        test_dataset = make_loader(vae_test, batch_size=batch_size, shuffle=False)

        stop = timer()
        self.preprocessing_time = (stop - start)

        start = timer()

        # VAE model and training
        self.vae = CVAE(input_shape=(vae_train.shape[1], vae_train.shape[2]), device=self.device)
        self.vae.fit(
            epochs=epochs,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            min_delta=min_delta,
            delta_patient=delta_patient,
            max_patient=max_patient,
            verbose=verbose,
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
            best = self.hof[0]
            self.best_dt = {
                "clf": best[0],
                "n_nodes": best[1],
                "n_leaves": best[2],
                "max_depth": best[3],
                "val_errors": best[4],
                "val_accuracy": best[5],
                "val_f1": best[6],
                "val_fitness_value": best[7]
            }
        else:  # GP didn't even find a real tree
            self.best_dt = None

        self.tot_is_tree = gp.tot_is_tree      # Number of times "isTree" has been called
        self.false_is_tree = gp.false_is_tree  # Number of times "isTree" returns False

        return self

    def predict(self, X):
        if self.is_fitted:
            return self.best_dt["clf"].predict(X)
        return None
