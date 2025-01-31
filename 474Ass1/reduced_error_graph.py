import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import copy



def reduced_error_pruning(tree, X_val, y_val, err):
    """
    Parameters:
    - tree: un-pruned decision tree
    - X_val: Data
    - y_val: Target

    Returns:
    - void: prunes tree using reduced error pruning

    I arbitrarily tested a few accuracies and looked at how it changed the tree.
    I'm not sure what this type of pruning is called, or if it has a name, or if
    it's fundamentally flawed. But I thought it was interesting that I could increase the accuracy
    of the tree this way.
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
        if post_accuracy < pre_accuracy * 1 - err:
            tree.tree_.children_left[node_index] = original_left
            tree.tree_.children_right[node_index] = original_right

    # Start pruning at the root
    prune(0)




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


trees = [copy.deepcopy(tree)] * 100
test_accuracies = []
train_accuracies = []
error_cutoffs = []

for i in range(100):
    reduced_error_pruning(trees[i], X_train, y_train, i*0.0001)
    y_pred = trees[i].predict(X_test)
    y_train_pred = trees[i].predict(X_train)
    test_accuracies.append(accuracy_score(y_test, y_pred))
    train_accuracies.append(accuracy_score(y_train, y_train_pred))
    error_cutoffs.append(i*0.0001)


plt.figure(figsize=(10, 5))
plt.plot(error_cutoffs[::10], train_accuracies[::10], marker='o', linestyle='-', label="Train Error")
plt.plot(error_cutoffs[::10], test_accuracies[::10], marker='o', linestyle='-', label="Test Error")
plt.xlabel("Alpha")
plt.ylabel("Accuracy")
plt.title("Alpha vs Accuracy (Reduced-Error Pruning)")
plt.legend()
plt.grid()

plt.show()

