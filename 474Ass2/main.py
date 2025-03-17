import numpy as np
import mnist_reader
from sklearn.preprocessing import normalize
import neural_networks
import svms
from matplotlib import pyplot as plt
import evaluation

datapath = 'fashion-mnist-master/data/fashion'

def add_label_noise(y, p = 0.269):
    flip_mask = np.random.rand(len(y)) < p
    y_noisy = y.copy()
    y_noisy[flip_mask] = 1 - y_noisy[flip_mask]

    return y_noisy

X_train, y_train = mnist_reader.load_mnist(datapath, kind='train')
X_test, y_test = mnist_reader.load_mnist(datapath, kind='t10k')

classes = [5, 7]

# Boolean mask to filter the dataset
train_mask = np.isin(y_train, classes)
test_mask = np.isin(y_test, classes)

# Apply the mask
X_train_binary = X_train[train_mask]
y_train_binary = y_train[train_mask]
X_test_binary = X_test[test_mask]
y_test_binary = y_test[test_mask]

X_train_binary = normalize(X_train_binary, norm='l2', axis=1)
X_test_binary = normalize(X_test_binary, norm='l2', axis=1)

# Convert labels: Class 5 → 0, Class 7 → 1
y_train_binary = np.where(y_train_binary == 5, 0, 1)
y_test_binary = np.where(y_test_binary == 5, 0, 1)

# Make noisy
y_train_binary = add_label_noise(y_train_binary)

X_train_binary_small = X_train_binary[:1000]
y_train_binary_small = y_train_binary[:1000]

# Print dataset shape
print(f"Training set shape: {X_train_binary.shape}, Labels shape: {y_train_binary.shape}")
print(f"Test set shape: {X_test_binary.shape}, Labels shape: {y_test_binary.shape}")

svm = svms.SVMClassifier(kernel="rbf", C=1.0, gamma="scale")
svm.train(X_train_binary, y_train_binary)
print("SVM Accuracy:", svm.accuracy(X_test_binary, y_test_binary))

svm = svms.SVMClassifier(kernel="linear")
svm.log_tune(X_train_binary_small, y_train_binary_small, B=1.5, C0=0.05, k=5)

C_values = [0.1 * (1.2 ** i) for i in range(30)]
test_scores = []
train_scores = []

best_score = -np.inf
for C in C_values:
        model = svms.SVMClassifier(kernel="linear", C=C)
        test_score = model.score(X_test_binary, y_test_binary)
        train_scores.append(model.score(X_train_binary, y_train_binary))
        test_scores.append(test_score)

        # Update best C
        if  test_score > best_score:
            best_score = test_score
            best_C = C
            best_linear_svm = model


fig, ax = plt.subplots()
ax.set_xlabel("Regularization Parameter (C)")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs C")
ax.plot(C_values, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(C_values, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()

print(f"\nBest C: {best_C:.5f} with accuracy {best_score:.4f}")

best_gaus_svm, best_gamma, best_gaus_C = svms.tune_gaussian_svm(X_train_binary_small, y_train_binary_small, X_test_binary, y_test_binary)

best_gaus_params = (best_gamma, best_gaus_C)

print(f"Best linear SVM score: {best_linear_svm.score(X_test, y_test)}")
print(f"Best gaussian SVM score: {best_gaus_svm.score(X_test, y_test)}")


hidden_layer_options = [(10,), (50,), (100,), (50, 50)]
activation_functions = ['relu', 'tanh']
epoch_values = [10, 50, 100, 200]

# Train and evaluate neural network
best_nn, results = neural_networks.train_neural_network(X_train_binary_small, y_train_binary_small, X_test_binary, y_test_binary, hidden_layer_options, activation_functions)

# Plot accuracy vs. hidden nodes
neural_networks.plot_hidden_nodes_vs_accuracy(results)

# Run controlled experiment on epochs
neural_networks.experiment_epochs(X_train_binary_small, y_train_binary_small, X_test_binary, y_test_binary, hidden_layers=(50,), activation='relu', epoch_values=epoch_values)

results = evaluation.evaluate_models(X_test_binary, y_test_binary, best_linear_svm, best_gaus_svm, best_nn)

evaluation.plot_test_errors(results)

