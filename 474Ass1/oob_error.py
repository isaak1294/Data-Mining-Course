import numpy as np
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load your dataset here (replace with actual data loading)
# X, Y = ... (features and labels)

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

# Parameters
n_estimators = 200  # Total number of trees
random_state = 42    # Fixed random state for reproducibility

# Train the Random Forest
rf = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=random_state,
    bootstrap=True,
    oob_score=False,  # We compute OOB manually
    warm_start=False
)
rf.fit(X, y)

# Get the number of samples
n_samples = X.shape[0]

# Collect OOB indices and predictions for each tree
unsampled_indices_for_all_trees = []
oob_predictions_per_tree = []

for tree in rf.estimators_:
    # Get OOB indices for the current tree
    random_instance = check_random_state(tree.random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    unsampled_indices = np.where(unsampled_mask)[0]
    unsampled_indices_for_all_trees.append(unsampled_indices)
    
    # Get predictions for OOB samples
    if len(unsampled_indices) > 0:
        X_oob = X[unsampled_indices]
        preds = tree.predict(X_oob)
    else:
        preds = np.array([])
    oob_predictions_per_tree.append((unsampled_indices, preds))

# Map each example to its OOB predictions and corresponding tree indices
example_oob_preds = [[] for _ in range(n_samples)]
for tree_idx, (oob_indices, preds) in enumerate(oob_predictions_per_tree):
    for i, example_idx in enumerate(oob_indices):
        example_oob_preds[example_idx].append((tree_idx, preds[i]))

# Compute OOB error for increasing numbers of trees
oob_errors = []
for t in range(1, n_estimators + 1):
    error_sum = 0.0
    n_valid_examples = 0
    
    for example_idx in range(n_samples):
        # Collect all predictions from trees < t where the example was OOB
        preds = []
        for (tree_idx, pred) in example_oob_preds[example_idx]:
            if tree_idx < t:
                preds.append(pred)
        
        if len(preds) == 0:
            continue  # Skip examples with no OOB predictions
        
        # Majority vote
        majority_vote = np.argmax(np.bincount(preds))
        if majority_vote != y[example_idx]:
            error_sum += 1
        n_valid_examples += 1
    
    if n_valid_examples > 0:
        oob_error = error_sum / n_valid_examples
    else:
        oob_error = 0.0  # Handle edge case with no OOB samples
    oob_errors.append(oob_error)

# Plot OOB error vs. number of trees
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_estimators + 1), oob_errors, marker='o', linestyle='-', markersize=3)
plt.xlabel('Number of Trees')
plt.ylabel('OOB Error')
plt.title('Out-of-Bag Error Estimate vs. Number of Trees')
plt.grid(True)
plt.show()