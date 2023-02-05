from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed
from utils import leaves_count, max_node_count
import random
import numpy as np
from functools import partial
from timeit import default_timer as timer

def return_depth(clf):
    return clf.tree_.max_depth

def return_node_count(clf):
    return clf.tree_.node_count

def return_linear_representation(clf, encoder, depths):
    max_depth = max(depths)
    return encoder.encode(clf=clf, depth=max_depth)

def generate_dt(X, y, max_depth, min_depth, random_prob, same_depth):
    clfs = []

    clf = DecisionTreeClassifier(max_depth=(max_depth if same_depth is True else random.randint(min_depth, max_depth)), 
                                random_state=random.randint(0, 2**32-1), 
                                splitter=("random" if random.random() < random_prob else "best"))
    
    clf = clf.fit(X, y)
    clfs.append(clf)

    return clfs

def dt_ensemble(X, y, max_depth, train_size, test_size, encoder, min_depth, random_prob, same_depth):

    # Train
    result = Parallel(n_jobs=-2)(delayed(generate_dt)(
                                                    X=X, 
                                                    y=y, 
                                                    max_depth=max_depth, 
                                                    min_depth=min_depth, 
                                                    random_prob=random_prob, 
                                                    same_depth=same_depth
                                                ) for _ in range(0, train_size))

    clfs = [tree[0] for tree in result] #new

    train_leaves = list(map(leaves_count, clfs))
    train_nodes = list(map(return_node_count, clfs))
    train_depths = list(map(return_depth, clfs))

    start = timer()
    train_trees = list(map(partial(return_linear_representation, encoder=encoder, depths=train_depths), clfs))
    stop = timer()
    train_encoding_time = (stop - start)

    # Test
    result = Parallel(n_jobs=-2)(delayed(generate_dt)(
                                                    X=X, 
                                                    y=y, 
                                                    max_depth=max_depth, 
                                                    min_depth=min_depth, 
                                                    random_prob=random_prob, 
                                                    same_depth=same_depth
                                                ) for _ in range(0, test_size))

    clfs = [tree[0] for tree in result] #new

    start = timer()
    test_trees = list(map(partial(return_linear_representation, encoder=encoder, depths=train_depths), clfs))
    stop = timer()
    test_encoding_time = (stop - start)
    encoding_time = train_encoding_time + test_encoding_time

    max_dim = max_node_count(max(train_depths))
    if (max_dim % 2) != 0:
        max_dim = max_dim + 1

    train_trees = [encoder.pad(vector=train_tree, dim=max_dim) for train_tree in train_trees if train_tree.shape[1]<max_dim]
    test_trees = [encoder.pad(vector=test_tree, dim=max_dim) for test_tree in test_trees if test_tree.shape[1]<max_dim]
    train = np.array(train_trees, dtype=object).astype('float32')
    test = np.array(test_trees, dtype=object).astype('float32')

    return [train, test, train_nodes, train_leaves, train_depths, encoding_time]

def new_rf(X, y, max_depth, train_size, test_size, encoder, min_depth, same_depth):

    nbr_trees_per_ensemble = 100
    nbr_rnd_forest = int(train_size/nbr_trees_per_ensemble) # 100
    n_samples = X.shape[0]
    min_samples = 100

    # Train
    clfs = []
    for _ in range(nbr_rnd_forest):
        clf_train = RandomForestClassifier(max_depth=(max_depth if same_depth is True else random.randint(min_depth, max_depth)),
                                        n_estimators=nbr_trees_per_ensemble,
                                        max_samples=random.randint(min_samples, n_samples)
                                    )
        clf_train = clf_train.fit(X, y)
        clfs.append(clf_train)

    clfs_train = [tree for forest in clfs for tree in forest]

    train_leaves = list(map(leaves_count, clfs_train))
    train_nodes = list(map(return_node_count, clfs_train))
    train_depths = list(map(return_depth, clfs_train))

    start = timer()
    train_trees = list(map(partial(return_linear_representation, encoder=encoder, depths=train_depths), clfs_train))
    stop = timer()
    train_encoding_time = (stop - start)

    nbr_rnd_forest = int(test_size/nbr_trees_per_ensemble) 

    # Test
    clfs = []
    for _ in range(nbr_rnd_forest):
        clf_test = RandomForestClassifier(max_depth=(max_depth if same_depth is True else random.randint(min_depth, max_depth)),
                                        n_estimators=nbr_trees_per_ensemble,
                                        max_samples=random.randint(min_samples, n_samples)
                                        )
        clf_test = clf_test.fit(X, y)
        clfs.append(clf_test)

    clfs_test = [tree for forest in clfs for tree in forest]

    start = timer()
    test_trees = list(map(partial(return_linear_representation, encoder=encoder, depths=train_depths), clfs_test))
    stop = timer()
    test_encoding_time = (stop - start)
    encoding_time = train_encoding_time + test_encoding_time

    max_dim = max_node_count(max(train_depths))
    if (max_dim % 2) != 0:
        max_dim = max_dim + 1

    train_trees = [encoder.pad(vector=train_tree, dim=max_dim) for train_tree in train_trees if train_tree.shape[1]<max_dim]
    test_trees = [encoder.pad(vector=test_tree, dim=max_dim) for test_tree in test_trees if test_tree.shape[1]<max_dim]
    train = np.array(train_trees, dtype=object).astype('float32')
    test = np.array(test_trees, dtype=object).astype('float32')

    return [train, test, train_nodes, train_leaves, train_depths, encoding_time]

def rf(X, y, max_depth, train_size, test_size, encoder):
    # Train
    clf_train = RandomForestClassifier(max_depth=max_depth,
                                       n_estimators=train_size,
                                       random_state=random.randint(0, 2**32-1))
    clf_train = clf_train.fit(X, y)

    train_leaves = list(map(leaves_count, clf_train.estimators_))
    train_nodes = list(map(return_node_count, clf_train.estimators_))
    train_depths = list(map(return_depth, clf_train.estimators_))

    start = timer()
    train_trees = list(map(partial(return_linear_representation, encoder=encoder, depths=train_depths), clf_train.estimators_))
    stop = timer()
    train_encoding_time = (stop - start)

    # Test
    clf_test = RandomForestClassifier(max_depth=max_depth,
                                      n_estimators=test_size,
                                      random_state=random.randint(0, 2**32-1))
    clf_test = clf_test.fit(X, y)

    start = timer()
    test_trees = list(map(partial(return_linear_representation, encoder=encoder, depths=train_depths), clf_test.estimators_))
    stop = timer()
    test_encoding_time = (stop - start)
    encoding_time = train_encoding_time + test_encoding_time

    max_dim = max_node_count(max(train_depths))
    if (max_dim % 2) != 0:
        max_dim = max_dim + 1

    train_trees = [encoder.pad(vector=train_tree, dim=max_dim) for train_tree in train_trees if train_tree.shape[1]<max_dim]
    test_trees = [encoder.pad(vector=test_tree, dim=max_dim) for test_tree in test_trees if test_tree.shape[1]<max_dim]
    train = np.array(train_trees, dtype=object).astype('float32')
    test = np.array(test_trees, dtype=object).astype('float32')

    return [train, test, train_nodes, train_leaves, train_depths, encoding_time]

def initial_population(
    X, 
    y, 
    max_depth, 
    train_size, 
    test_size, 
    encoder, 
    mode="DT", 
    min_depth=1, 
    random_prob=0.3, 
    same_depth=False
): 
    if mode == "RF":
        return rf(X, y, max_depth, train_size, test_size, encoder)
    elif mode == "DT":
        return dt_ensemble(X, y, max_depth, train_size, test_size, encoder, min_depth, random_prob, same_depth)   
    elif mode == "NEW_RF":
        return new_rf(X, y, max_depth, train_size, test_size, encoder, min_depth, same_depth)
    else:
        print("Error: value '{}' for 'mode' is not valid. Possible values are 'DT', 'RF' and 'NEW_RF'.".format(mode))