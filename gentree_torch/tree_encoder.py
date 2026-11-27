import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree

from .utils import get_depth, is_leaf, max_node_count, prune_duplicate_leaves


class JakowskiEncoder:
    """Linear (2 x n_nodes) encoding of a decision tree.

    Row 0: feature index + 1 for internal nodes, -1 for leaves, 0 for padding.
    Row 1: threshold for internal nodes, class label + 1 for leaves.
    Node ``i`` has children ``2i+1`` (left) and ``2i+2`` (right).
    """

    def __init__(self, n_features, n_classes):
        self.n_features_in_ = n_features
        self.n_classes_ = n_classes
        self.classes_ = np.arange(n_classes)
        self.n_outputs_ = 1

    def encode(self, clf, depth):
        decisions = clf.tree_.value.argmax(axis=2).flatten().tolist()

        vector = np.zeros((2, max_node_count(depth)))

        index = {0: 0}  # {node : node index in the vector}
        parent = {}     # {node : parent_node}

        for node in range(clf.tree_.node_count):
            if is_leaf(clf.tree_, node):
                vector[0][index[node]] = -1
                vector[1][index[node]] = decisions[node] + 1  # class label
            else:
                vector[0][index[node]] = clf.tree_.feature[node] + 1     # feature [1, m]
                vector[1][index[node]] = clf.tree_.threshold[node]       # threshold

                index[clf.tree_.children_left[node]] = 2 * index[node] + 1
                index[clf.tree_.children_right[node]] = 2 * index[node] + 2

                parent[2 * index[node] + 1] = index[node]
                parent[2 * index[node] + 2] = index[node]

        # Push every leaf down to the last level (the node takes its parent's split)
        for node in range(vector.shape[1]):
            if vector[0][node] == -1:
                if ((2 * node + 1) < (vector.shape[1] - 1)) and ((2 * node + 2) < vector.shape[1]):
                    vector[0][2 * node + 1] = -1
                    vector[1][2 * node + 1] = vector[1][node]
                    vector[0][2 * node + 2] = -1
                    vector[1][2 * node + 2] = vector[1][node]

                    parent[2 * node + 1] = node
                    parent[2 * node + 2] = node

                    vector[0][node] = vector[0][parent[node]]
                    vector[1][node] = vector[1][parent[node]]

        return vector

    @staticmethod
    def _reachable(vector):
        # Vector columns reachable from the root (pre-order, same order used by decode)
        order, stack = [], [0]
        while stack:
            index = stack.pop()
            order.append(index)
            if vector[0][index] != -1:
                stack.append(2 * index + 2)
                stack.append(2 * index + 1)
        return order

    def decode(self, vector):
        # Count only the nodes reachable from the root: nodes hanging below a leaf
        # are ignored by the traversal and must not be allocated in the Tree.
        node_count = len(self._reachable(vector))
        n_classes_ = np.array([self.n_classes_], dtype=np.intp)

        inner_tree = Tree(self.n_features_in_, n_classes_, self.n_outputs_)  # underlying Tree object
        state = inner_tree.__getstate__()

        state['node_count'] = node_count
        # Use the node dtype of the installed scikit-learn (>=1.3 adds 'missing_go_to_left')
        nodes = np.zeros((node_count,), dtype=state['nodes'].dtype)
        values = np.zeros((node_count, self.n_outputs_, self.n_classes_))

        internal_nodes = []  # array of internal nodes (index of the node in the tree)
        current_id = 0

        stack = [0]  # root
        children = {}  # {vector column index: (0-L/1-R, index of the parent node in the tree)}

        while stack:
            index = stack.pop()

            # Save the tree index of the child in the parent node
            if index != 0:
                side, parent_id = children[index]
                nodes[parent_id]['right_child' if side else 'left_child'] = current_id

            if vector[0][index] == -1:  # leaf
                nodes[current_id]['left_child'] = -1
                nodes[current_id]['right_child'] = -1
                nodes[current_id]['feature'] = -2
                nodes[current_id]['threshold'] = -2.0
                nodes[current_id]['n_node_samples'] = 1
                nodes[current_id]['weighted_n_node_samples'] = 1.0
                values[current_id][0][int(vector[1][index] - 1)] = 1.0
            else:  # internal node
                internal_nodes.append(current_id)

                nodes[current_id]['feature'] = vector[0][index] - 1    # feature [0, m-1]
                nodes[current_id]['threshold'] = vector[1][index]      # threshold

                children[2 * index + 2] = (1, current_id)
                stack.append(2 * index + 2)
                children[2 * index + 1] = (0, current_id)
                stack.append(2 * index + 1)

            current_id += 1

        # Update the number of samples belonging to each internal node
        for node in internal_nodes[::-1]:
            left, right = nodes[node]['left_child'], nodes[node]['right_child']
            nodes[node]['n_node_samples'] = nodes[left]['n_node_samples'] + nodes[right]['n_node_samples']
            nodes[node]['weighted_n_node_samples'] = (nodes[left]['weighted_n_node_samples'] +
                                                      nodes[right]['weighted_n_node_samples'])
            values[node][0] = values[left][0] + values[right][0]
            nodes[node]['impurity'] = 1 - sum((values[node][0] / sum(values[node][0])) ** 2)

        state['nodes'] = nodes
        state['values'] = values
        inner_tree.__setstate__(state)

        state = inner_tree.__getstate__()
        state['max_depth'] = get_depth(inner_tree)
        inner_tree.__setstate__(state)

        clf = DecisionTreeClassifier()
        clf.tree_ = inner_tree
        clf.n_outputs_ = inner_tree.n_outputs
        clf.n_classes_ = inner_tree.n_classes[0]
        clf.classes_ = np.arange(inner_tree.n_classes[0])
        clf.n_features_in_ = inner_tree.n_features
        clf.max_features_ = inner_tree.n_features

        prune_duplicate_leaves(clf)

        return clf

    def is_tree(self, vector):
        # Checks if vector matches a tree.

        # At least 3 nodes (1 internal_node and 2 leaf_node)
        if vector.shape[1] < 3:
            return False

        # n_leaves = n_internal_nodes + 1
        n_internal_nodes = len(np.where(vector[0] > 0)[0].tolist())
        n_leaves = vector[0].flatten().tolist().count(-1)

        if n_internal_nodes < 1:
            return False

        if n_leaves != (n_internal_nodes + 1):
            return False

        # Features (first row, non-leaf columns) must be in [1, n_features]
        features = set(int(x) for x in vector[0][vector[0] != -1])
        if min(features) < 1 or max(features) > self.n_features_in_:
            return False

        classes = set(int(x) for x in vector[1][vector[0] == -1])
        if min(classes) <= 0 or max(classes) > self.n_classes_:
            return False

        # The last internal node's right child must be the last column
        last_internal_node = np.where(vector[0] > 0)[0].tolist()[-1]

        if (last_internal_node * 2 + 2) != (vector.shape[1] - 1):
            return False

        return True

    def pad(self, vector, dim, constant_values=0):
        if dim > vector.shape[1]:
            diff = dim - vector.shape[1]
            return np.pad(array=vector, pad_width=((0, 0), (0, diff)), mode='constant',
                          constant_values=constant_values)
        return vector

    def unpad(self, vector):
        index = vector.shape[1]
        for i in range(index - 1, 0, -1):
            if vector[0][i] != 0 and vector[1][i] != 0:
                index = i + 1
                break
        return vector[:, :index]

    def complete_tree(self, clf, X):
        # Assigns the instances of X to each node of the DT.
        # See https://ieeexplore.ieee.org/abstract/document/8242347

        internal_nodes = np.where(clf.tree_.feature >= 0)[0].tolist()  # IDs of internal nodes
        internal_nodes.sort(reverse=True)

        leaves = clf.apply(X)     # leaf index of each sample
        classes = clf.predict(X)  # predicted class of each sample

        state = clf.tree_.__getstate__()
        nodes, values = state['nodes'], state['values']

        for node in range(clf.tree_.node_count):
            nodes[node]['n_node_samples'] = 0
            nodes[node]['weighted_n_node_samples'] = 0
            values[node][0] = np.zeros(clf.n_classes_,)

        for i in range(len(leaves)):
            values[leaves[i]][0][int(classes[i])] += 1
            nodes[leaves[i]]['n_node_samples'] += 1
            nodes[leaves[i]]['weighted_n_node_samples'] += 1

        for node in internal_nodes:
            left, right = nodes[node]['left_child'], nodes[node]['right_child']
            nodes[node]['n_node_samples'] = nodes[left]['n_node_samples'] + nodes[right]['n_node_samples']
            nodes[node]['weighted_n_node_samples'] = (nodes[left]['weighted_n_node_samples'] +
                                                      nodes[right]['weighted_n_node_samples'])
            values[node][0] = values[left][0] + values[right][0]
            nodes[node]['impurity'] = 1 - sum((values[node][0] / sum(values[node][0])) ** 2)

        clf.tree_.__setstate__(state)
