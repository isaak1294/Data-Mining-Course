import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# Define hyperparameter ranges
n_estimators_range = range(1, 201, 10)  # Number of trees
max_depth_range = range(1, 21)  # Max depth

# Store error rates
errors_n_estimators = []
train_errors_n_estimators = []
errors_max_depth = []
train_errors_max_depth = []

# Vary Number of Trees (`n_estimators`)
for n in n_estimators_range:
    clf = RandomForestClassifier(n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    y_train_pred = clf.predict(X_train)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    errors_n_estimators.append(error)
    train_errors_n_estimators.append(train_error)

# Vary Max Depth (`max_depth`)
for depth in max_depth_range:
    clf = RandomForestClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)
    error = 1 - accuracy_score(y_test, y_pred)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    errors_max_depth.append(error)
    train_errors_max_depth.append(train_error)

# Make a random forest
rf_classifier = RandomForestClassifier(
    n_estimators=100,       # Size of the ensemble
    max_depth=None,         # Maximum depth of the trees (None = unlimited depth)
    random_state=69         # Seed for reproducibility
)

# Train random forest
rf_classifier.fit(X_train, y_train)

# Make a prediction on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plotting

feature_importances = rf_classifier.feature_importances_
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances')


# Number of Trees vs Error
plt.figure(figsize=(10, 5))
plt.plot(n_estimators_range, errors_n_estimators, marker='o', linestyle='-', color='b', label="Test Error")

plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Error Rate")
plt.title("Number of Trees vs. Error Rate")
plt.legend()
plt.grid()


# Max Depth vs Error
plt.figure(figsize=(10, 5))
plt.plot(max_depth_range, errors_max_depth, marker='s', linestyle='-', color='r', label="Test Error")

plt.xlabel("Max Depth")
plt.ylabel("Error Rate")
plt.title("Max Depth vs. Error Rate")
plt.legend()
plt.grid()
plt.show()