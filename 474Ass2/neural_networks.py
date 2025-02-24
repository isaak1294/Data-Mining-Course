import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize a simple neural network.

        Parameters:
        - input_size: Number of input features (e.g., 784 for Fashion-MNIST)
        - hidden_size: Number of neurons in the hidden layer
        - output_size: Number of output classes (2 for binary classification)
        """
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Raw logits (cross-entropy loss will handle softmax)

def train_neural_net(model, X_train, y_train, epochs=10, lr=0.001):
    """
    Train a neural network.

    Parameters:
    - model: The neural network model.
    - X_train: Training features (PyTorch tensor).
    - y_train: Training labels (PyTorch tensor).
    - epochs: Number of training epochs.
    - lr: Learning rate.
    """
    criterion = nn.CrossEntropyLoss()  # Handles multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def evaluate_neural_net(model, X_test, y_test):
    """ Evaluate the neural network's accuracy. """
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_test).float().mean()
    return accuracy.item()
