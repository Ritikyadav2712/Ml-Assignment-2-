import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load Dataset
independent_file = "logisticX.csv" # Replace with the actual file path
dependent_file = "logisticY.csv" # Replace with the actual file path

# Load datasets
X = pd.read_csv(independent_file, header=None)
y = pd.read_csv(dependent_file, header=None).values.ravel()

# Display the first few rows to confirm successful loading
print("Independent Variables (X):")
print(X.head())
print("\nDependent Variable (y):")
print(y[:5])

# Initialize the logistic regression model
model = LogisticRegression(solver='lbfgs', max_iter=1000)

# Train the model
model.fit(X, y)

# Retrieve model coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression Cost Function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1 / m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    return cost

# Gradient Descent Implementation
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    costs = []
    for _ in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradient
        costs.append(compute_cost(X, y, theta))
    return theta, costs

# Add bias term (intercept)
X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones for intercept
theta_init = np.zeros(X_bias.shape[1])  # Initialize theta

# Perform gradient descent
learning_rate = 0.1
iterations = 50
theta, costs = gradient_descent(X_bias, y, theta_init, learning_rate, iterations)

# Plot the cost function
plt.plot(range(iterations), costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function vs. Iterations")
plt.show()

# Plot the dataset and decision boundary
plt.scatter(X[0], X[1], c=y, cmap='viridis')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Dataset and Decision Boundary")

x_values = np.linspace(X[0].min(), X[0].max(), 100)
y_values = -(theta[1] * x_values + theta[0]) / theta[2]
plt.plot(x_values, y_values, color='red', label="Decision Boundary")
plt.legend()
plt.show()

# Polynomial Feature Augmentation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Train logistic regression on the new dataset
model.fit(X_poly, y)

# Plot new dataset and decision boundary
plt.scatter(X[0], X[1], c=y, cmap='viridis')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Polynomial Features and Decision Boundary")

# Generate grid for plotting decision boundary
x1_vals = np.linspace(X[0].min(), X[0].max(), 100)
x2_vals = np.linspace(X[1].min(), X[1].max(), 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Predict values for the grid
Z = model.predict(poly.transform(np.c_[X1.ravel(), X2.ravel()]))
Z = Z.reshape(X1.shape)

plt.contour(X1, X2, Z, levels=[0.5], colors='red')
plt.show()

# Make predictions using the polynomially transformed dataset
y_pred = model.predict(X_poly)

# Confusion matrix
conf_matrix = confusion_matrix(y, y_pred)

# Calculate metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

