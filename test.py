import numpy as np
import pandas as pd

# Example data (e.g., some time series)
y = np.array([4.0, 4.5, 5.0, 5.3, 5.7, 6.1, 6.4, 6.8])

p = 2  # AR(2) model

# Create lagged dataset
df = pd.DataFrame({'y': y})
for i in range(1, p+1):
    df[f'y_lag{i}'] = df['y'].shift(i)

df = df.dropna()

# Prepare X (lags) and y (current values)
X = df[['y_lag1', 'y_lag2']].values
Y = df['y'].values

# Add intercept term for c
X = np.column_stack((np.ones(len(X)), X))

# Solve for coefficients using OLS: beta = (X'X)^(-1) X'Y
beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)

c = beta[0]        # intercept
phi_1 = beta[1]    # phi_1
phi_2 = beta[2]    # phi_2

print(f"Intercept (c): {c}")
print(f"phi_1: {phi_1}")
print(f"phi_2: {phi_2}")
