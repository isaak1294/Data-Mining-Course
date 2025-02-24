from sklearn.svm import SVC
import k_fold

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
        best_C = C0
        best_score = 0
        
        # Perform k-fold cross-validation for each C
        for C in C_values:
            model = SVC(kernel=self.model.kernel, C=C, gamma=self.model.gamma)
            scores, mean_score = k_fold.k_fold(X, y, model, k)

            print(f"C = {C:.5f}, Cross-validation Accuracy: {mean_score:.4f}")

            # Update best C
            if mean_score > best_score:
                best_score = mean_score
                best_C = C

        print(f"\nBest C: {best_C:.5f} with accuracy {best_score:.4f}")

        # Train the final model using the best C
        self.model = SVC(kernel=self.model.kernel, C=best_C, gamma=self.model.gamma)
        self.model.fit(X, y)




