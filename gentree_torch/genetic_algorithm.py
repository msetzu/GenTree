import random
from math import inf

import numpy as np
import torch
from deap import base, creator, tools
from sklearn.metrics import accuracy_score, f1_score

# Created at import time to avoid DEAP warnings when the GA runs more than once
# (guarded so that importing both the TF and the torch versions does not clash).
if not hasattr(creator, "Fitness"):
    creator.create("Fitness", base.Fitness, weights=(-1.0,))  # -1: minimize fitness
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.Fitness)  # individual = [latent_vector]


def data_postprocessing(data):
    # Rounds feature ids (row 0) and, for leaves and padding, class labels (row 1). In-place.
    data[0] = np.around(data[0, :]).astype('int')
    to_round = (data[0] == -1) | (data[0] == 0)
    data[1][to_round] = np.around(data[1][to_round])


def fitness_a(clf, X, y, omega=0.75, is_tree=True):
    if is_tree:
        not_correct_tree = 0
        n_leaves = clf.tree_.threshold.tolist().count(-2)
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
    else:
        not_correct_tree = inf
        n_leaves = inf
        accuracy = 0

    return (1 - accuracy) + omega * (n_leaves / len(y)) + not_correct_tree


def fitness_b(clf, X, y, omega=0.75, is_tree=True):
    if is_tree:
        n_nodes = clf.tree_.node_count
        n_leaves = clf.tree_.threshold.tolist().count(-2)
        n_internal_nodes = n_nodes - n_leaves

        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)

        return 1 - (omega * (1 / (n_internal_nodes + 1)) + (1 - omega) * accuracy)
    else:
        return 1


def fitness_c(clf, X, y, omega=0.75, is_tree=True):
    if is_tree:
        not_correct_tree = 0
        n_leaves = clf.tree_.threshold.tolist().count(-2)
        y_pred = clf.predict(X)
        f1 = f1_score(y_true=y, y_pred=y_pred, average='weighted')
    else:
        not_correct_tree = inf
        n_leaves = inf
        f1 = 0

    return (1 - f1) + omega * (n_leaves / len(y)) + not_correct_tree


def fitness_d(clf, X, y, omega=0.75, is_tree=True):
    if is_tree:
        n_nodes = clf.tree_.node_count
        n_leaves = clf.tree_.threshold.tolist().count(-2)
        n_internal_nodes = n_nodes - n_leaves

        y_pred = clf.predict(X)
        f1 = f1_score(y_true=y, y_pred=y_pred, average='weighted')

        return 1 - (omega * (1 / (n_internal_nodes + 1)) + (1 - omega) * f1)
    else:
        return 1


class GeneticProcess:
    def __init__(
        self,
        model,
        scalers,
        encoder,
        max_depth,
        X_test,
        y_test,
        fitness,
        initial_tree=None,
        mutation="custom",
        omega=0.75,
        ngen=50,
        population_size=100,
        cx_pb=0.6,
        mut_pb=0.001,
        tournsize=3,
        plot=True
    ):
        self.model = model
        self.scalers = scalers
        self.encoder = encoder
        self.max_depth = max_depth
        self.X_test = X_test
        self.y_test = y_test
        self.initial_tree = initial_tree
        self.mutation = mutation
        self.fitness = fitness
        self.omega = omega
        self.ngen = ngen                            # number of generations
        self.population_size = population_size      # number of individuals per generation
        self.cx_pb = cx_pb                          # crossover probability
        self.mut_pb = mut_pb                        # mutation probability
        self.tournsize = tournsize
        self.n_hof = ngen                           # size of DEAP's Hall of Fame
        self.plot = plot
        self.tot_is_tree = 0
        self.false_is_tree = 0

    # ------------------------------------------------------------------ #
    # Latent <-> tree
    # ------------------------------------------------------------------ #
    def _latent_to_vectors(self, latents):
        """Decodes a batch of latent vectors into unpadded (2, n_nodes) tree encodings."""
        z = np.asarray(latents, dtype=np.float32).reshape(-1, self.model.latent_dim)
        with torch.no_grad():
            decoded = self.model.decode(z, apply_sigmoid=True).cpu().numpy()  # (n, length, 2)

        for i in range(decoded.shape[2]):
            decoded[:, :, i] = self.scalers[i].inverse_transform(decoded[:, :, i])

        decoded = np.transpose(decoded, (0, 2, 1))  # (n, 2, length)

        vectors = []
        for i in range(decoded.shape[0]):
            data_postprocessing(decoded[i])
            vectors.append(self.encoder.unpad(vector=decoded[i]))
        return vectors

    def _fitness(self, clf, is_tree=True):
        return self.fitness(clf=clf, X=self.X_test, y=self.y_test, omega=self.omega, is_tree=is_tree)

    def _evaluate_vector(self, vector):
        self.tot_is_tree += 1
        if self.encoder.is_tree(vector=vector):
            fitness_value = self._fitness(self.encoder.decode(vector=vector))
        else:
            fitness_value = self._fitness(None, is_tree=False)
            self.false_is_tree += 1
        return fitness_value,  # DEAP fitness must be a tuple

    def _evaluate(self, individual):
        return self._evaluate_vector(self._latent_to_vectors([individual[0]])[0])

    def _evaluate_population(self, individuals):
        # Batched version of `_evaluate` (one decoder call for all individuals)
        if not individuals:
            return
        vectors = self._latent_to_vectors([ind[0] for ind in individuals])
        for ind, vector in zip(individuals, vectors):
            ind.fitness.values = self._evaluate_vector(vector)

    # ------------------------------------------------------------------ #
    # Operators
    # ------------------------------------------------------------------ #
    def _random_individual(self):
        # Returns the latent vector (list) of a new individual
        if self.initial_tree is not None:
            length = self.model._input_shape[0]
            tree = self.encoder.encode(clf=self.initial_tree, depth=self.max_depth)
            tree = self.encoder.pad(vector=tree, dim=length)            # (2, length)
            tree = np.transpose(tree[np.newaxis], (0, 2, 1)).astype('float32')  # (1, length, 2)

            # Same scaling as the VAE training data
            for i in range(tree.shape[2]):
                tree[:, :, i] = self.scalers[i].transform(tree[:, :, i])

            with torch.no_grad():
                tree_mean, tree_logvar = self.model.encode(tree)
                z = self.model.reparameterize(tree_mean, tree_logvar)
            return z.cpu().numpy()[0].tolist()
        else:
            return self.model.latent_space().cpu().numpy()[0].tolist()

    def _mutate(self, individual):
        # Replaces the latent vector (in place) with a new random point of the latent space
        individual[:] = self.model.latent_space().cpu().numpy()[0].tolist()
        return individual,

    def _setup(self):
        self.tot_is_tree = 0    # counts how many times "encoder.is_tree" is called
        self.false_is_tree = 0  # counts how many times "encoder.is_tree" returns false

        self.toolbox = base.Toolbox()
        self.toolbox.register("random_individual", self._random_individual)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.random_individual, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)

        if self.mutation == "custom":
            self.toolbox.register("mutate", self._mutate)
        elif self.mutation == "gaussian":
            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.8)
        else:
            raise ValueError("'mutation' must be in ['custom', 'gaussian']")

        self.toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

    def run(self):
        self._setup()

        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(self.n_hof)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('min', np.min)
        stats.register('avg', np.mean)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        invalid_individuals = [ind for ind in pop if not ind.fitness.valid]
        self._evaluate_population(invalid_individuals)
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(invalid_individuals), **record)
        print(logbook.stream)

        for gen in range(1, self.ngen + 1):
            # Select and clone the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_pb:
                    self.toolbox.mate(child1[0], child2[0])
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mut_pb:
                    self.toolbox.mutate(mutant[0])
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self._evaluate_population(invalid_ind)

            hof.update(offspring)
            pop[:] = offspring

            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

        hof.update(pop)

        self.logbook = logbook
        self.best_solutions = hof.items

        if self.plot:
            import matplotlib.pyplot as plt
            import seaborn as sns

            minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
            plt.figure()
            sns.set_style("whitegrid")
            plt.plot(minFitnessValues, color='blue')
            plt.plot(meanFitnessValues, color='green')
            plt.xlabel('Generation')
            plt.ylabel('Fitness Value')
            plt.title('Avg and Min Fitness')
            plt.show()

    def hof(self, n=10):
        n = min(n, self.n_hof)

        hof = []
        best_fitness = None

        if not self.best_solutions:
            return None

        vectors = self._latent_to_vectors([ind[0] for ind in self.best_solutions])

        for vector in vectors:
            if self.encoder.is_tree(vector=vector):
                clf = self.encoder.decode(vector=vector)

                n_leaves = clf.tree_.threshold.tolist().count(-2)
                n_nodes = clf.tree_.node_count

                y_pred = clf.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(y_true=self.y_test, y_pred=y_pred, average='weighted')
                fitness_value = self._fitness(clf)
                errors = int(np.sum(y_pred != self.y_test))

                individual = [clf, n_nodes, n_leaves, clf.tree_.max_depth, errors, accuracy, f1, fitness_value]

                if best_fitness is None or best_fitness > fitness_value:
                    best_fitness = fitness_value
                    hof.insert(0, individual)
                else:
                    hof.append(individual)

        if not hof:  # GP didn't even find a real tree
            return None

        hof.sort(key=lambda ind: ind[-1])  # sort by 'fitness_value'
        return hof[:n]
