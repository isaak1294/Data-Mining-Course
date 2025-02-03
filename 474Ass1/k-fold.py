import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def k_fold(X, y, model, k=5, shuffle=True, random_state=69):
    """
    Parameters:
    - X (ndarray): Data
    - y (ndarray): Target
    - model: a model to perform the validation on
    - k (int): number of folds
    - shuffle (bool): whether or not to shuffle the data
    - random_state (int): parameter to pass to random_seed

    Returns:
    - list: A list of the accuracy scores of each fold
    - float: The average accuracy across all folds
    """
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
    else:
        indices = np.arange(len(X))

    # Split indices into k folds:
    fold_sizes = len(X)//k
    fold_accuracies = []

    for i in range(k):
        # Define the test fold
        start = i * fold_sizes
        end = start + fold_sizes if i != k-1 else len(X)
        test_indices = indices[start:end]

        # Define the train fold
        train_indices = np.concatenate([indices[:start], indices[end:]])

        # Split data
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train the model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)

    return fold_accuracies, np.mean(fold_accuracies)


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

# Fixed hyperparameters
max_features = int(np.sqrt(X.shape[1]))  # sqrt(d)
criterion = "gini"
max_depth = None

base_estimator = DecisionTreeClassifier(max_depth=1)  # Decision stump
learning_rate = 1.0  # Default learning rate

# Ensemble sizes to evaluate
n_trees_range = [10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

# Perform k-fold cross-validation for each ensemble size
k = 5  # Number of folds
cv_scores = []
ada_cv_scores = []

for n_trees in n_trees_range:
    clf = RandomForestClassifier(
        n_estimators=n_trees,
        max_features=max_features,
        criterion=criterion,
        max_depth=max_depth,
        random_state=42
    )
    fold_scores, avg_score = k_fold(X, y, clf)
    cv_scores.append(avg_score)


    ada = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_trees,
        learning_rate=learning_rate,
        random_state=42
    )
    fold_scores, avg_score = k_fold(X, y, clf)
    ada_cv_scores.append(avg_score)



# Plot results
plt.figure(figsize=(8, 4))
plt.plot(n_trees_range, cv_scores, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Random Forest: Ensemble Size Tuning')
plt.grid(True)
plt.savefig('figures/ensemble_size_tuning.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(8, 4))
plt.plot(n_trees_range, ada_cv_scores, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Stumps (n_estimators)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('AdaBoost: Ensemble Size Tuning')
plt.grid(True)
plt.savefig('figures/ada_ensemble_size_tuning.png', dpi=300, bbox_inches='tight')
plt.show()

# Select optimal ensemble size
optimal_n_trees = n_trees_range[np.argmax(cv_scores)]
print(f"Optimal number of trees: {optimal_n_trees}")





