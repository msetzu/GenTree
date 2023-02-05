from deap import base, creator, tools
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from math import inf

# Here to avoid a warning that appears when the genetic algorithm runs more than once
creator.create("Fitness", base.Fitness, weights=(-1.0,)) # <- -1 because we want to minimize fitness
creator.create("Individual", list, fitness=creator.Fitness) # <- the individual is defined as a list

def data_postprocessing(data):
    data[0] = np.around(data[0, :]).astype('int')
    leaves = np.where(data[0] == -1)
    padding = np.where(data[0] == 0)

    for j in range(0, data.shape[1]):
        if len(leaves[0]) != 0:
            if j in leaves[0]:
                data[1][j] = np.around(data[1][j])
        if len(padding[0]) != 0:   
            if j in padding[0]:
                data[1][j] = np.around(data[1][j])

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

    return (1 - accuracy) + omega*(n_leaves/len(y)) + not_correct_tree

def fitness_b(clf, X, y, omega=0.75, is_tree=True):
    if is_tree:
        n_nodes = clf.tree_.node_count
        n_leaves = clf.tree_.threshold.tolist().count(-2)
        n_internal_nodes = n_nodes - n_leaves

        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        return 1 - (omega*(1/(n_internal_nodes + 1)) + (1 - omega)*accuracy)
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

    return (1 - f1) + omega*(n_leaves/len(y)) + not_correct_tree

def fitness_d(clf, X, y, omega=0.75, is_tree=True):
    if is_tree:
        n_nodes = clf.tree_.node_count
        n_leaves = clf.tree_.threshold.tolist().count(-2)
        n_internal_nodes = n_nodes - n_leaves

        y_pred = clf.predict(X)
        f1 = f1_score(y_true=y, y_pred=y_pred, average='weighted')
        
        return 1 - (omega*(1/(n_internal_nodes + 1)) + (1 - omega)*f1)
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
        self.population_size = population_size      # number of individual for generation
        self.cx_pb = cx_pb                          # crossover probability
        self.mut_pb = mut_pb                        # mutation probability
        self.tournsize = tournsize
        self.n_hof = ngen                           # top solutions to return (DEAP's "Hall of Fame" is the set of all top n solutions)
        self.plot = plot

    def __random_individual(self):  
        # Returns a list of values. An individual is a list.
        # latent_space() returns -> tf.Tensor: shape=(1, latent_dim), dtype=float32, numpy=array([[a, b, c]])

        if self.initial_tree is not None:
            tree = self.encoder.encode(clf=self.initial_tree, depth=self.max_depth)

            if self.model._input_shape[0] > tree.shape[0]:
                tree = self.encoder.pad(vector=tree, dim=self.model._input_shape[0])

            tree = np.reshape(tree, (1, tree.shape[0], tree.shape[1]))
            tree = np.transpose(tree, (0, 2, 1))

            tree_mean, tree_logvar = self.model.encode(tree)
            
            eps = tf.random.normal(shape=tree_mean.shape)
            tree_latent_representation = eps * tf.exp(tree_logvar * .5) + tree_mean
            tree_latent_representation = tree_latent_representation.numpy()
            
            return tree_latent_representation[0].tolist()

        else:
            return self.model.latent_space().numpy().tolist()[0]

    def __evaluate(self, individual):
        # A typical evaluation function takes one individual as argument and returns its fitness as a tuple
        
        individual_copy = np.copy(individual[0]) # faccio la copia dell'individuo
        
        individual_copy = tf.convert_to_tensor([individual_copy]) # from list to tensor
        individual_copy = self.model.decode(individual_copy, apply_sigmoid=True).numpy() 
        
        for i in range(individual_copy.shape[2]):
            individual_copy[:, :, i] = self.scalers[i].inverse_transform(individual_copy[:, :, i]) 
        
        individual_copy = np.transpose(individual_copy, (0, 2, 1))

        for i in range(individual_copy.shape[0]):
            data_postprocessing(individual_copy[i])
            
        individual_copy = individual_copy[0]

        unpadded_vec = self.encoder.unpad(vector=individual_copy) 
        decoded_vec = None    

        self.tot_is_tree = self.tot_is_tree + 1

        if self.encoder.is_tree(vector=unpadded_vec):
            decoded_vec = self.encoder.decode(vector=unpadded_vec)
            fitness_value = self.fitness(clf=decoded_vec, X=self.X_test, y=self.y_test)
        else:
            fitness_value = self.fitness(clf=decoded_vec, X=self.X_test, y=self.y_test, is_tree=False)
            self.false_is_tree = self.false_is_tree + 1

        return fitness_value, # evaluate must always return a tuple (there's why the comma)

    def __mutate(self, individual, tree=None):
        individual = None
        
        individual = self.model.latent_space()
        
        return individual.numpy().tolist()[0],

    def __setup(self):
        #creator.create("Fitness", base.Fitness, weights=(-1.0,)) 
        #creator.create("Individual", list, fitness=creator.Fitness) 
        self.tot_is_tree = 0    # counts how many times "encoder.is_tree" is called
        self.false_is_tree = 0  # counts how many times "encoder.is_tree" returns false

        self.toolbox = base.Toolbox() #creiamo il toolbox

        self.toolbox.register("random_individual", self.__random_individual) 

        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.random_individual, n=1) 

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.__evaluate) 
        self.toolbox.register("mate", tools.cxTwoPoint) #funzione di crossover

        if self.mutation == "custom":
            self.toolbox.register("mutate", self.__mutate) 
        elif self.mutation == "gaussian":
            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.8)
        else:
            print("[ERROR]: 'mutation' in ['custom', 'gaussian']")

        self.toolbox.register("select", tools.selTournament, tournsize=self.tournsize) #tournsize=3

    def run(self):
        self.__setup()

        pop = self.toolbox.population(n=self.population_size)

        hof = tools.HallOfFame(self.n_hof)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)   
        stats.register('min', np.min) 
        stats.register('avg', np.mean) 
        
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields
        
        invalid_individuals = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit
            
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=0, best="-", nevals=len(invalid_individuals), **record)
        print(logbook.stream)
        
        for gen in range(1, self.ngen + 1):
            
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
                  
            # Apply crossover and mutation on the offspring
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
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        
            # Update the hall of fame with the generated individuals
            hof.update(offspring)
            
            # Replace the current population by the offspring
            pop[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(pop) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)
            
            
        hof.update(pop) 
        
        plt.figure(1)

        minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
        self.best_solutions = hof.items

        if self.plot:
            plt.figure(2)
            sns.set_style("whitegrid")
            plt.plot(minFitnessValues, color='blue')
            plt.plot(meanFitnessValues, color='green')
            plt.xlabel('Generation')
            plt.ylabel('Fitness Value')
            plt.title('Avg and Min Fitness')
            # show both plots:
            plt.show()

    def hof(self, n=10):
        if n > self.n_hof:
            n = self.n_hof

        hof = []
        best_fitness = None

        for j in range(len(self.best_solutions)):
            clf = tf.convert_to_tensor(self.best_solutions[j]) 
            clf = self.model.decode(clf, apply_sigmoid=True).numpy()

            for i in range(clf.shape[2]):
                clf[:, :, i] = self.scalers[i].inverse_transform(clf[:, :, i])

            clf = np.transpose(clf, (0, 2, 1))

            for i in range(clf.shape[0]):
                data_postprocessing(clf[i])

            clf = clf[0]

            clf = self.encoder.unpad(vector=clf)

            if self.encoder.is_tree(vector=clf):
                clf = self.encoder.decode(vector=clf) 

                n_leaves = clf.tree_.threshold.tolist().count(-2)
                n_nodes = clf.tree_.node_count

                y_pred = clf.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(y_true=self.y_test, y_pred=y_pred, average='weighted')
                fitness_value = self.fitness(clf=clf, X=self.X_test, y=self.y_test)
                errors = sum(map(lambda x,y: bool(x-y), y_pred, self.y_test))

                individual = [clf, n_nodes, n_leaves, clf.tree_.max_depth, errors, accuracy, f1, fitness_value]

                if best_fitness is None or best_fitness > fitness_value:
                    best_fitness = fitness_value
                    hof.insert(0, individual)
                else:
                    hof.append(individual)

        hof = np.array(hof)

        if hof.ndim <= 1: # GP doesn't even find a real tree
            return None
        else:
            hof = hof[hof[:, -1].argsort()] # sort hof by 'fitness_value'
            return hof[:n].tolist()
