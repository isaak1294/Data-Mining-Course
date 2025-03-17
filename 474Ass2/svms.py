from sklearn.svm import SVC
import k_fold
import matplotlib.pyplot as plt

class SVMClassifier:
    def __init__(self, kernel="linear", C=1.0, gamma="scale"):
        """
        Initialize an SVM classifier.

        Parameters:
        - kernel: "linear" for linear SVM, "rbf" for Gaussian SVM.
        - C: Regularization parameter (higher C means less regularization).
        - gamma: Kernel coefficient for RBF kernel. Can be 'scale', 'auto', or a float.
        """
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)

    def train(self, X_train, y_train):
        """ Train the SVM model """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """ Predict using the trained SVM """
        return self.model.predict(X)

    def accuracy(self, X, y):
        """ Compute accuracy """
        return (self.predict(X) == y).mean()
    
    def log_tune(self, X, y, B=2, C0=1, k=5):
        """
        Perform hyperparameter tuning for C using a logarithmic grid.

        Parameters:
        - X, y: Training data and labels.
        - B: Base for logarithmic scaling.
        - C0: Initial value of C.
        - k: Number of folds in cross-validation.
        """
        # Generate a set of 10 logarithmically spaced C values
        C_values = [C0 * (B ** i) for i in range(10)]
        C_scores = []
        best_C = C0
        best_score = 0
        
        # Perform k-fold cross-validation for each C
        for C in C_values:
            model = SVC(kernel=self.model.kernel, C=C, gamma=self.model.gamma)
            scores, mean_score = k_fold.k_fold(X, y, model, k)

            print(f"C = {C:.5f}, Cross-validation Accuracy: {mean_score:.4f}")

            C_scores.append(mean_score.round(2))

            # Update best C
            if mean_score > best_score:
                best_score = mean_score
                best_C = C

        print(f"\nBest C: {best_C:.5f} with accuracy {best_score:.4f}")

        # Generate cross-validation score vs C value graph
        plt.figure(figsize=(8, 5))
        plt.plot(C_values, C_scores, marker='o', linestyle='-')
        plt.xscale('log')  # Log scale for better visualization
        plt.xlabel("Regularization Parameter (C)")
        plt.ylabel("Cross-Validation Accuracy")
        plt.title("Cross-Validation Accuracy vs. C")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

        # Train the final model using the best C
        self.model = SVC(kernel=self.model.kernel, C=best_C, gamma=self.model.gamma)
        self.model.fit(X, y)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

def tune_gaussian_svm(X_train, y_train, X_test, y_test, B=2, gamma0=1e-3, C0=1, k=5):
    """
    Tune hyperparameters C and gamma for an SVM with Gaussian kernel.
    """
    # Define a logarithmically spaced set of gamma values
    gamma_values = [gamma0 * (B ** i) for i in range(10)]
    best_gamma, best_C, best_score = None, None, 0
    gamma_scores = []
    
    # Find the best (gamma, C_gamma) pair
    for gamma in gamma_values:
        C_values = [C0 * (B ** i) for i in range(10)]
        best_C_gamma, best_score_gamma = C0, 0
        
        for C in C_values:
            model = SVC(kernel='rbf', C=C, gamma=gamma)
            scores = cross_val_score(model, X_train, y_train, cv=k)
            mean_score = scores.mean()
            
            if mean_score > best_score_gamma:
                best_score_gamma = mean_score
                best_C_gamma = C
        
        gamma_scores.append(best_score_gamma)
        if best_score_gamma > best_score:
            best_score = best_score_gamma
            best_gamma, best_C = gamma, best_C_gamma
    
    print(f"Best gamma: {best_gamma:.5f}, Best C: {best_C:.5f}, Best Cross-validation Accuracy: {best_score:.4f}")
    
    # Train final model with best parameters
    final_model = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
    final_model.fit(X_train, y_train)
    
    # Evaluate on training and test sets
    train_error = 1 - accuracy_score(y_train, final_model.predict(X_train))
    test_error = 1 - accuracy_score(y_test, final_model.predict(X_test))
    
    # Plot Cross-validation Accuracy vs Gamma
    plt.figure(figsize=(8, 5))
    plt.semilogx(gamma_values, gamma_scores, marker='o', linestyle='-')
    plt.xlabel("Gamma (log scale)")
    plt.ylabel("Cross-validation Accuracy")
    plt.title("Cross-validation Accuracy vs Gamma")
    plt.grid(True)
    plt.show()
    
    # Plot Training & Test Error vs Gamma
    plt.figure(figsize=(8, 5))
    plt.semilogx(gamma_values, [train_error] * len(gamma_values), label="Training Error", linestyle='--')
    plt.semilogx(gamma_values, [test_error] * len(gamma_values), label="Test Error", linestyle='-')
    plt.xlabel("Gamma (log scale)")
    plt.ylabel("Error")
    plt.title("Training & Test Error vs Gamma")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return final_model, best_gamma, best_C


