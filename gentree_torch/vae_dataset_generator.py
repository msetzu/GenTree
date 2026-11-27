import random
from functools import partial
from timeit import default_timer as timer

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .utils import leaves_count, max_node_count


def return_depth(clf):
    return clf.tree_.max_depth


def return_node_count(clf):
    return clf.tree_.node_count


def return_linear_representation(clf, encoder, depths):
    max_depth = max(depths)
    return encoder.encode(clf=clf, depth=max_depth)


def generate_dt(X, y, max_depth, min_depth, random_prob, same_depth, seed=None):
    # `seed` is drawn in the parent process so that runs are reproducible with joblib workers
    rng = random.Random(seed) if seed is not None else random
    clf = DecisionTreeClassifier(max_depth=(max_depth if same_depth is True else rng.randint(min_depth, max_depth)),
                                 random_state=rng.randint(0, 2**32 - 1),
                                 splitter=("random" if rng.random() < random_prob else "best"))
    clf = clf.fit(X, y)
    return [clf]


def _encode_and_pad(train_clfs, test_clfs, encoder):
    """Encodes train/test trees at the max train depth and pads them to an even length.

    Returns (train, test, train_nodes, train_leaves, train_depths, encoding_time).
    """
    train_leaves = list(map(leaves_count, train_clfs))
    train_nodes = list(map(return_node_count, train_clfs))
    train_depths = list(map(return_depth, train_clfs))
    max_depth = max(train_depths)

    start = timer()
    train_trees = list(map(partial(return_linear_representation, encoder=encoder, depths=train_depths), train_clfs))
    stop = timer()
    train_encoding_time = (stop - start)

    # Test trees deeper than every training tree cannot be encoded in the same space
    test_clfs = [clf for clf in test_clfs if clf.tree_.max_depth <= max_depth]

    start = timer()
    test_trees = list(map(partial(return_linear_representation, encoder=encoder, depths=train_depths), test_clfs))
    stop = timer()
    test_encoding_time = (stop - start)
    encoding_time = train_encoding_time + test_encoding_time

    max_dim = max_node_count(max_depth)
    if (max_dim % 2) != 0:
        max_dim = max_dim + 1

    train_trees = [encoder.pad(vector=t, dim=max_dim) for t in train_trees if t.shape[1] < max_dim]
    test_trees = [encoder.pad(vector=t, dim=max_dim) for t in test_trees if t.shape[1] < max_dim]
    train = np.array(train_trees, dtype=object).astype('float32')
    test = np.array(test_trees, dtype=object).astype('float32')

    return [train, test, train_nodes, train_leaves, train_depths, encoding_time]


def dt_ensemble(X, y, max_depth, train_size, test_size, encoder, min_depth, random_prob, same_depth):
    def _generate(n):
        result = Parallel(n_jobs=-2)(delayed(generate_dt)(
            X=X, y=y, max_depth=max_depth, min_depth=min_depth,
            random_prob=random_prob, same_depth=same_depth,
            seed=random.randint(0, 2**32 - 1)) for _ in range(0, n))
        return [tree[0] for tree in result]

    return _encode_and_pad(_generate(train_size), _generate(test_size), encoder)


def new_rf(X, y, max_depth, train_size, test_size, encoder, min_depth, same_depth):
    nbr_trees_per_ensemble = 100
    n_samples = X.shape[0]
    min_samples = min(100, n_samples)

    def _generate(n):
        clfs = []
        for _ in range(int(n / nbr_trees_per_ensemble)):
            clf = RandomForestClassifier(max_depth=(max_depth if same_depth is True else random.randint(min_depth, max_depth)),
                                         n_estimators=nbr_trees_per_ensemble,
                                         max_samples=random.randint(min_samples, n_samples))
            clfs.append(clf.fit(X, y))
        return [tree for forest in clfs for tree in forest]

    return _encode_and_pad(_generate(train_size), _generate(test_size), encoder)


def rf(X, y, max_depth, train_size, test_size, encoder):
    def _generate(n):
        clf = RandomForestClassifier(max_depth=max_depth,
                                     n_estimators=n,
                                     random_state=random.randint(0, 2**32 - 1))
        return clf.fit(X, y).estimators_

    return _encode_and_pad(_generate(train_size), _generate(test_size), encoder)


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
        raise ValueError("Value '{}' for 'mode' is not valid. Possible values are 'DT', 'RF' and 'NEW_RF'.".format(mode))
