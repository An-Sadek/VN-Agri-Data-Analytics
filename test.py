import numpy as np

# --- shapes:
# n_state = 2 (e.g., displacement, velocity)
# n_meas  = 2 (we measure both)
# n_feat  = k (number of exogenous features)

n_state = 2
n_meas = 2
n_feat = 3  # example: 3 features

# --- Initialize state and matrices
x = np.array([30.0, 20.0])          # (n_state,)
A = np.array([[1., 1.],
              [0., 1.]])            # (n_state, n_state)

H = np.eye(n_meas, n_state)         # (n_meas, n_state)
Q = np.array([[0.004, 0.002],
              [0.002, 0.001]])      # (n_state, n_state)
R = np.array([[0.4, 0.01],
              [0.04, 0.01]])        # (n_meas, n_meas)
P = np.zeros((n_state, n_state))    # (n_state, n_state)

# --- D: measurement-feature mapping (n_meas x n_feat)
# You can initialize D from prior knowledge or estimate it from historical data (OLS).
D = np.random.normal(scale=0.1, size=(n_meas, n_feat))

# Example data (replace with your arrays)
total_time = len(measurement)
measurements = np.array(measurement)            # shape (T, n_meas)
features = np.array(exog_features)              # shape (T, n_feat)

estimates = []

for t in range(total_time):
    z = measurements[t]      # (n_meas,)
    f = features[t]          # (n_feat,)

    # -- Prediction
    x = A @ x
    P = A @ P @ A.T + Q

    # -- Update with feature regression term
    z_pred = H @ x + D @ f   # predicted measurement including features
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    y = z - z_pred           # innovation (residual)
    x = x + K @ y
    P = (np.eye(n_state) - K @ H) @ P

    estimates.append(x.copy())

estimates = np.array(estimates)
