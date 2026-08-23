from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load dataset
data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df2 = pd.DataFrame(data.target, columns=["price"])

X = df
y = df2["price"]

print(X.shape)
print(y.shape)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Feature scaling
scale = StandardScaler()

# Learn mean and standard deviation from training data
scale.fit(X_train)

# Apply scaling
X_train_scaled = scale.transform(X_train)
X_test_scaled = scale.transform(X_test)


print("Mean:")
print(X_train_scaled.mean(axis=0))

print("\nStandard deviation:")
print(X_train_scaled.std(axis=0))


# Initialize weights and bias
w = np.zeros(X_train_scaled.shape[1])
b = 0

learning_rate = 0.01
iterations = 1000

m = len(y_train)
n=len(w)

# Store cost values to see how cost changes
cost_history = []

lambda_=1

# Gradient descent
for i in range(iterations):

    # Prediction
    y_predict = X_train_scaled @ w + b

    # Error
    error = y_predict - y_train

    # Cost function
    cost = np.mean(error ** 2) / 2
    
    #regularization cost
    reg_cost=0
    for j in range(n):
        reg_cost+= (lambda_ / (2*m)) * (w[j]**2)
     # Total regularized cost
    total_cost = cost + reg_cost
    cost_history.append(total_cost)

    # Gradients
    normal_dw = (X_train_scaled.T @ error) / m 
    #regularization graidant 
    reg_dw = (lambda_ / m) * w
    dw = normal_dw + reg_dw
    db = np.mean(error) # common for both normal and regualrization

    # Update weights and bias
    w = w - learning_rate * dw
    b = b - learning_rate * db


print("Final weights:")
print(w)

print("Final bias:")
print(b)


# Training predictions
y_train_pred = X_train_scaled @ w + b

print("Predicted:")
print(y_train_pred[:10])

print("Actual:")
print(y_train.iloc[:10].values)
#Train Predictions
error= y_train_pred-y_train
mse=np.mean(error**2)
rmse=np.sqrt(mse)
train_r2 = 1 - (
    np.sum((y_train - y_train_pred) ** 2)
    / np.sum((y_train - np.mean(y_train)) ** 2))
print("Train mse:", mse)
print("Train rmse:", rmse)
print("Train R²:", train_r2)

# Test predictions
y_test_pred = X_test_scaled @ w + b

print("Test predicted:")
print(y_test_pred[:10])

print("Test actual:")
print(y_test.iloc[:10].values)


# Calculate MSE
error = y_test_pred - y_test
mse = np.mean(error ** 2)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate R²
r2 = 1 - (
    np.sum((y_test - y_test_pred) ** 2)
    / np.sum((y_test - np.mean(y_test)) ** 2)
)

print("MSE:", mse)
print("RMSE:", rmse)
print("Model R²:", r2)



# Predict a new house
new_house = pd.DataFrame([[
    5.0,       # MedInc
    30.0,      # HouseAge
    5.5,       # AveRooms
    1.0,       # AveBedrms
    1000.0,    # Population
    2.5,       # AveOccup
    34.0,      # Latitude
    -118.0     # Longitude
]], columns=X_train.columns)

# Scale using the training scaler
new_house_scaled = scale.transform(new_house)

# Predict price
new_house_prediction = new_house_scaled @ w + b

print("Predicted price:", new_house_prediction[0])


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(y_test.values[:50], label="Actual", marker="o")
plt.plot(y_test_pred[:50], label="Predicted", marker="x")

plt.xlabel("Test Sample")
plt.ylabel("House Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()

plt.show()






    


