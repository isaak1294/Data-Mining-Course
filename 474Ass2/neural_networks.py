import numpy as np
import matplotlib.pyplot as plt
import k_fold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def train_neural_network(X_train, y_train, X_test, y_test, hidden_layer_options, activation_functions, k=5, epochs=100):
    """
    Train and tune a neural network using k-fold cross-validation.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Test data and labels.
    - hidden_layer_options: List of different hidden layer configurations to try.
    - activation_functions: List of activation functions to experiment with.
    - k: Number of folds in cross-validation.
    - epochs: Number of training epochs.

    Returns:
    - Best trained model.
    """

    best_model = None
    best_score = 0
    best_params = None
    results = []


    for hidden_layers in hidden_layer_options:
        for activation in activation_functions:
            model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, max_iter=epochs, random_state=42, n_jobs=-1)

            scores, mean_score = k_fold.k_fold(X_train, y_train, model, k)

            results.append((hidden_layers, activation, mean_score))

            print(f"Hidden Layers: {hidden_layers}, Activation: {activation}, CV Accuracy: {mean_score:.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_params = (hidden_layers, activation)

    print(f"\nBest Model: Hidden Layers {best_params[0]}, Activation {best_params[1]}, Accuracy {best_score:.4f}")

    # Train final model on full training data
    best_model.fit(X_train, y_train)

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy of Best Model: {test_accuracy:.4f}")

    return best_model, results

def plot_hidden_nodes_vs_accuracy(results):
    """Plot cross-validation accuracy vs. number of hidden nodes."""
    hidden_nodes = [sum(layers) for layers, _, _ in results]
    accuracies = [score for _, _, score in results]

    plt.figure(figsize=(8, 5))
    plt.plot(hidden_nodes, accuracies, marker='o', linestyle='--', color='b')
    plt.xscale("log")
    plt.xlabel("Number of Hidden Nodes")
    plt.ylabel("Cross-validation Accuracy")
    plt.title("Cross-validation Accuracy vs. Number of Hidden Nodes")
    plt.grid()
    plt.show()

def experiment_epochs(X_train, y_train, X_test, y_test, hidden_layers, activation, epoch_values):
    """Vary number of epochs and plot training/test error."""
    train_errors = []
    test_errors = []

    for epochs in epoch_values:
        model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, max_iter=epochs, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_error = 1 - accuracy_score(y_train, y_train_pred)
        test_error = 1 - accuracy_score(y_test, y_test_pred)

        train_errors.append(train_error)
        test_errors.append(test_error)

        print(f"Epochs: {epochs}, Training Error: {train_error:.4f}, Test Error: {test_error:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_values, train_errors, label="Training Error", marker='o', linestyle='--', color='r')
    plt.plot(epoch_values, test_errors, label="Test Error", marker='o', linestyle='--', color='b')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Error")
    plt.title("Training & Test Error vs. Epochs")
    plt.legend()
    plt.grid()
    plt.show()