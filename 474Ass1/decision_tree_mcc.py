## Data Mining - Fall 2023 (author: Nishant Mehta)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import check_random_state  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



# Retreive the data from the csv
data = np.genfromtxt('spambase_augmented.csv', delimiter=',', skip_header=0, filling_values=np.nan)

# Target the last column (whether the email was spam or not)
X = data[:, :-1]
y = data[:, -1]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training sets 80/20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state=69)

# Make a decision tree
tree = DecisionTreeClassifier().fit(X_train, y_train)

path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

trees = []
for alpha in ccp_alphas:
    t = DecisionTreeClassifier(ccp_alpha=alpha)
    t.fit(X_train, y_train)
    trees.append(t)

print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        trees[-1].tree_.node_count, ccp_alphas[-1]
    )
)

trees = trees[:-1]
ccp_alphas = ccp_alphas[:-1]
"""
    node_counts = [tree.tree_.node_count for tree in trees]
    depth = [tree.tree_.max_depth for tree in trees]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()

    plt.show()
"""

# Make prediction?

train_scores = [clf.score(X_train, y_train) for clf in trees]
test_scores = [clf.score(X_test, y_test) for clf in trees]

feature_names = [f"Feature {i}" for i in range(X.shape[1])]  # Generic feature names
class_names = [str(cls) for cls in np.unique(y)]  # Unique class labels as strings

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha (minimum cost complexity pruning)")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()

for i, t in enumerate(test_scores):
    print(f"index: {i}, score: {t}")

plt.figure(figsize=(12, 8))
plot_tree(trees[10], filled=True, feature_names=feature_names, class_names=class_names)

plt.show()


## So i think that i made a decision tree with minimal cost complexity pruning here?