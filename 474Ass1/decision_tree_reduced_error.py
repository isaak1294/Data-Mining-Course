import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import copy


def calculate_gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts/counts.sum()
    return 1 - np.sum(probabilities ** 2)

def calculate_split_gini(y_left, y_right):
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_right + n_left

    gini_left = calculate_gini(y_left)
    gini_right = calculate_gini(y_right)

    weighted_gini = (n_left/n_total) * gini_left + (n_right/n_total) * gini_right
    return weighted_gini


def reduced_error_pruning(tree, X_val, y_val):
    """
    Parameters:
    - tree: un-pruned decision tree
    - X_val: Data
    - y_val: Target

    Returns:
    - void: prunes tree using reduced error pruning
    """
    # Recursive pruning function:
    def prune(node_index):
        if not hasattr(tree.tree_, "children_left"):
            return
        
        # Get left and right child indices:
        left_child = tree.tree_.children_left[node_index]
        right_child = tree.tree_.children_right[node_index]

        # If node is a leaf, return
        if left_child == -1 and right_child == -1:
            return
        
        # Prune left and right trees recursively
        if left_child != -1:
            prune(left_child)
        if right_child != -1:
            prune(right_child)

        original_left = tree.tree_.children_left[node_index]
        original_right = tree.tree_.children_right[node_index]

        # Get the pre-pruning accuracy for comparison
        pre_prediction = tree.predict(X_val)
        pre_accuracy = accuracy_score(y_val, pre_prediction)

        # Make the current node a leaf
        tree.tree_.children_left[node_index] = -1
        tree.tree_.children_right[node_index] = -1

        # Get the post-pruning accuracy for comparison
        post_prediction = tree.predict(X_val)
        post_accuracy = accuracy_score(y_val, post_prediction)

        # If the tree got less accurate, go back
        if post_accuracy < pre_accuracy:
            tree.tree_.children_left[node_index] = original_left
            tree.tree_.children_right[node_index] = original_right

    # Start pruning at the root
    prune(0)

def prune_with_gini(tree, node_id, alpha=0.01):
    """
    Prunes a decision tree using a Gini-based criterion with a gain threshold.

    Parameters:
    - tree: The tree structure (e.g., clf.tree_ from scikit-learn).
    - node_id: The current node index.
    - alpha: Minimum impurity gain required to retain a split (prune if gain < alpha).
    
    Returns:
    - bool: True if pruned, False otherwise.
    """
    # If it's a leaf, nothing to prune
    if tree.children_left[node_id] == -1 and tree.children_right[node_id] == -1:
        return False

    left_id = tree.children_left[node_id]
    right_id = tree.children_right[node_id]

    # Recursively prune children first (post-order traversal)
    pruned_left = prune_with_gini(tree, left_id, alpha)
    pruned_right = prune_with_gini(tree, right_id, alpha)

    # Calculate Gini impurity gain for the current node
    pre_gini = tree.impurity[node_id]
    n_left = tree.n_node_samples[left_id]
    n_right = tree.n_node_samples[right_id]
    total = n_left + n_right

    # Use updated child impurities (after potential pruning)
    post_gini = (n_left / total) * tree.impurity[left_id] + (n_right / total) * tree.impurity[right_id]
    gain = pre_gini - post_gini  # How much the split improved impurity

    # Prune only if the gain is below the threshold
    if gain < alpha:
        # Convert this node into a leaf
        tree.children_left[node_id] = -1
        tree.children_right[node_id] = -1
        tree.n_node_samples[node_id] = total
        tree.impurity[node_id] = post_gini
        return True  # Node was pruned

    return pruned_left or pruned_right

# Retreive the data from the csv
data = np.genfromtxt('spambase_augmented.csv', delimiter=',', skip_header=0, filling_values=np.nan)

# Target the last column (whether the email was spam or not)
X = data[:, :-1]
y = data[:, -1]

feature_names = [f"Feature {i}" for i in range(X.shape[1])]  # Generic feature names
class_names = [str(cls) for cls in np.unique(y)]  # Unique class labels as strings

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training sets 80/20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state=69)

# Initialize tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

copy1 = copy.deepcopy(tree)
copy2 = copy.deepcopy(tree)

# Get the accuracy before for analysis
accuracy_before = accuracy_score(y_test, tree.predict(X_test))
print(f"Accuracy before pruning: {accuracy_before:.2f}")

# Do the pruning
reduced_error_pruning(copy1, X_train, y_train)

# Get the accuracy after pruning
accuracy_after = accuracy_score(y_test, copy1.predict(X_test))
print(f"Accuracy after reduced error pruning, no split criterion: {accuracy_after:.2f}")

prune_with_gini(copy2.tree_, 0)

accuracy_after = accuracy_score(y_test, copy2.predict(X_test))
print(f"Accuracy after reduced error pruning with gini index: {accuracy_after:.2f}")

plt.figure(figsize=(12, 8))
plot_tree(tree, filled=True, feature_names=feature_names, class_names=class_names)

plt.figure(figsize=(12, 8))
plot_tree(copy1, filled=True, feature_names=feature_names, class_names=class_names)

plt.figure(figsize=(12, 8))
plot_tree(copy2, filled=True, feature_names=feature_names, class_names=class_names)

plt.show()