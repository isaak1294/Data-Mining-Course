import numpy as np
import mnist_reader
from sklearn.preprocessing import normalize
import neural_networks
import svms
import torch

datapath = 'fashion-mnist-master/data/fashion'

def add_label_noise(y, p = 0.269):
    flip_mask = np.random.rand(len(y)) < p
    y_noisy = y.copy()
    y_noisy[flip_mask] = 1 - y_noisy[flip_mask]

    return y

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

# Print dataset shape
print(f"Training set shape: {X_train_binary.shape}, Labels shape: {y_train_binary.shape}")
print(f"Test set shape: {X_test_binary.shape}, Labels shape: {y_test_binary.shape}")

svm = svms.SVMClassifier(kernel="rbf", C=1.0, gamma="scale")
svm.train(X_train_binary, y_train_binary)
print("SVM Accuracy:", svm.accuracy(X_test_binary, y_test_binary))

# Neural Network Training
input_size = 784  # Fashion-MNIST flattened images
hidden_size = 128  # Can be tuned
output_size = 2  # Binary classification (classes 5 and 7)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_binary, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_binary, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_binary, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_binary, dtype=torch.long)

# Initialize and train
nn_model = neural_networks.NeuralNet(input_size, hidden_size, output_size)
neural_networks.train_neural_net(nn_model, X_train_tensor, y_train_tensor, epochs=20, lr=0.001)

# Evaluate
nn_accuracy = neural_networks.evaluate_neural_net(nn_model, X_test_tensor, y_test_tensor)
print("Neural Network Accuracy:", nn_accuracy)

svm = svms.SVMClassifier(kernel="linear")
svm.log_tune(X_train_binary, y_train_binary, B=2, C0=0.01, k=5)