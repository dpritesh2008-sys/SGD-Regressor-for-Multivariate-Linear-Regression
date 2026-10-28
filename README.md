# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Create the input and output datasets.
3. Split the data into training and testing sets.
4. Standardize the data.
5. Train the model using `SGDRegressor`.
6. Predict the output values.
7. Calculate Mean Squared Error (MSE).
8. Plot actual and predicted values.
9. Display the results and stop.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Ritesh DP
RegisterNumber:  212225040339
*/
```

```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data = fetch_california_housing()

# Select first 3 features (for demonstration)
X = data.data[:, :3]   # shape (n_samples, 3)

# Create a multi-output target: [median_house_value, some_other_numeric_column]
# Here we use column index 6 (for demonstration) as the second output
Y = np.column_stack((data.target, data.data[:, 6]))

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Example X (first row):", X[0])
print("Example Y (first row):", Y[0])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Train shapes:", X_train.shape, Y_train.shape)
print("Test shapes: ", X_test.shape, Y_test.shape)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Fit on training data and transform both train and test
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

print("Scaled X_train mean (approx):", X_train_scaled.mean(axis=0))
print("Scaled Y_train mean (approx):", Y_train_scaled.mean(axis=0))
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)  # you can also set alpha, eta0, penalty etc.
multi_output_sgd = MultiOutputRegressor(sgd)

# Fit on scaled training data
multi_output_sgd.fit(X_train_scaled, Y_train_scaled)
Y_pred_scaled = multi_output_sgd.predict(X_test_scaled)   # predicted in scaled space
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)         # back to original units
Y_test_orig = scaler_Y.inverse_transform(Y_test_scaled)    # ground-truth back to original

print("First 5 predictions (original units):")
print(Y_pred[:5])
mse = mean_squared_error(Y_test_orig, Y_pred)
print("Mean Squared Error (multi-output):", mse)

# Per-output MSE (optional, helpful for debugging)
mse_per_output = np.mean((Y_test_orig - Y_pred) ** 2, axis=0)
print("MSE per output:", mse_per_output)
for i in range(5):
    print(f"Example {i+1}")
    print("Inputs (raw):", X_test[i])
    print("True outputs:", Y_test_orig[i])
    print("Predicted   :", Y_pred[i])
    print("-" * 40)
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = fetch_california_housing()
X, y = data.data[:, :3], data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# SGD Regressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)

# Compare
print("LinearRegression MSE:", mean_squared_error(y_test, lr_pred))
print("SGDRegressor MSE:", mean_squared_error(y_test, sgd_pred))
```
## Output:
<img width="500" height="86" alt="image" src="https://github.com/user-attachments/assets/348d7de0-93dc-4a98-8dd2-d18e29364a1d" />
<img width="324" height="41" alt="image" src="https://github.com/user-attachments/assets/278e9e03-f7f0-47d2-a144-6d50d0c2de85" />
<img width="632" height="42" alt="image" src="https://github.com/user-attachments/assets/639b3a47-545f-4fe6-b5d5-736b25f90ac4" />
<img width="217" height="142" alt="image" src="https://github.com/user-attachments/assets/bbfd8d9f-e738-430e-91a3-cd2f5cebd8d9" />
<img width="473" height="161" alt="image" src="https://github.com/user-attachments/assets/9a71d447-7ddc-4c04-a1d1-ac6408d83b99" />
<img width="310" height="142" alt="image" src="https://github.com/user-attachments/assets/dd326550-2dac-4afb-bc23-f3c09f2e4acd" />
<img width="435" height="60" alt="image" src="https://github.com/user-attachments/assets/f76d0c02-42c4-47fb-b232-0c2434b7a53e" />
<img width="430" height="470" alt="image" src="https://github.com/user-attachments/assets/b1a9e508-8483-44ba-82b7-2fd6bb309f65" />
<img width="321" height="45" alt="image" src="https://github.com/user-attachments/assets/90bd9578-573c-42cb-bbfa-a347da20faeb" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
