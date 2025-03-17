import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import norm

def compute_confidence_interval(accuracy, n, confidence=0.95):
    """
    Compute the confidence interval for classification accuracy.

    Parameters:
    - accuracy: Model accuracy (between 0 and 1).
    - n: Number of test samples.
    - confidence: Confidence level (default 95%).

    Returns:
    - (lower bound, upper bound) of confidence interval.
    """
    error = 1 - accuracy
    z_score = norm.ppf((1 + confidence) / 2)
    std_error = np.sqrt((error * (1 - error)) / n)
    margin = z_score * std_error
    return max(0, error - margin), min(1, error + margin)

def evaluate_models(X_test, y_test, linear_svm, gaussian_svm, neural_net):
    """
    Train and compare the optimally tuned models.

    Returns:
    - Dictionary with model names, test errors, and confidence intervals.
    """
    models = ["Linear SVM", "Gaussian SVM", "Neural Network"]
    test_errors = []
    error_bars = []

    for model, classifier in zip(models, [linear_svm, gaussian_svm, neural_net]):
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        ci = compute_confidence_interval(acc, len(y_test))

        test_errors.append(1 - acc)  # Convert accuracy to error
        error_bars.append([ci[0], ci[1]])  # Store confidence intervals

        print(f"{model}: Test Error = {1 - acc:.4f}, Confidence Interval = ({ci[0]:.4f}, {ci[1]:.4f})")

    return {
        "models": models,
        "test_errors": test_errors,
        "error_bars": error_bars
    }


def plot_test_errors(results):
    """Plot test errors for all models."""
    models = results['models']
    errors = results['test_errors']
    ci_lowers = [results['error_bars'][i][0] for i in range(len(models))]
    ci_uppers = [results['error_bars'][i][1] for i in range(len(models))]

    # Compute yerr as the difference from the error to the confidence interval
    yerr = [(errors[i] - ci_lowers[i], ci_uppers[i] - errors[i]) for i in range(len(models))]

    plt.figure(figsize=(8, 5))
    plt.bar(models, errors, color=['blue', 'green', 'red'], yerr=np.array(yerr).T, capsize=5)
    plt.ylabel("Test Error")
    plt.title("Final Comparison: Test Errors of All Models")
    plt.grid(axis='y')
    plt.show()
