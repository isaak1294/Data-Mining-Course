import numpy as np
from sklearn.metrics import accuracy_score

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
        np.random_seed(random_state)
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
        train_indices = np.concatenate(indices[:start], indices[end:])

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








