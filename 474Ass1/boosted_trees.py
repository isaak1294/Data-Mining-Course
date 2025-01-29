import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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

# Initialize a base estimator
base_estimator = DecisionTreeClassifier(
    max_depth=1,            # Use shallow trees (stumps)
    random_state=42
)

# Initialize adaboost
adaboost_classifier = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,        # Size of the ensemble
    learning_rate=1.0,      # Shrinks the contribution of each classifier
    random_state=42
)

# Define hyperparameter ranges
n_estimators_range = range(1, 201, 10)  # Number of trees
learning_rate_range = range(1, 10)  # Max depth

# Store error rates
errors_n_estimators = []
errors_learning_rate = []

for n in n_estimators_range:
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), 
                             n_estimators=n, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    errors_n_estimators.append(error)

for l in learning_rate_range:
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
                              n_estimators=50, learning_rate=l*0.1, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    errors_learning_rate.append(error)

# Train ada
adaboost_classifier.fit(X_train, y_train)

# Make preditions
y_pred = adaboost_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Graphing
feature_importances = adaboost_classifier.feature_importances_
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importances')

# Number of Trees vs Error
plt.figure(figsize=(10, 5))
plt.plot(n_estimators_range, errors_n_estimators, marker='o', linestyle='-', color='b', label="Test Error")
plt.xlabel("Number of Rounds (n_estimators)")
plt.ylabel("Error Rate")
plt.title("Number of Trees vs. Error Rate")
plt.legend()
plt.grid()

plt.show()