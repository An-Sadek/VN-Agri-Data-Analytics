import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Simulate 2D time series
np.random.seed(42)
t = np.arange(100)
A = 10 + np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.3, size=100)
B = 20 + np.cos(2 * np.pi * t / 24) + np.random.normal(0, 0.3, size=100)

# Inject outliers
A[[30, 70]] += 4
B[[40, 85]] -= 5

df = pd.DataFrame({'A': A, 'B': B})
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

from sklearn.ensemble import IsolationForest

# Fit model
iforest = IsolationForest(contamination=0.05, random_state=42)
labels_iforest = iforest.fit_predict(X_scaled)  # -1 = outlier, 1 = inlier

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[labels_iforest == 1, 0], X_scaled[labels_iforest == 1, 1], label="Normal", c="green", s=40)
plt.scatter(X_scaled[labels_iforest == -1, 0], X_scaled[labels_iforest == -1, 1], label="Outlier", c="red", s=40)
plt.title("Isolation Forest - Outlier Detection")
plt.xlabel("A (scaled)")
plt.ylabel("B (scaled)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
