import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# --- 1. Simulate multivariate data ---
np.random.seed(42)
t = np.arange(100)
A = 10 + np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.3, size=100)
B = 20 + np.cos(2 * np.pi * t / 24) + np.random.normal(0, 0.3, size=100)

# Inject anomalies
A[[30, 70]] += 4
B[[40, 85]] -= 5

df = pd.DataFrame({'A': A, 'B': B})

# --- 2. Scale the data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['A', 'B']])

# --- 3. Train One-Class SVM ---
model = OneClassSVM(kernel="rbf", gamma='scale', nu=0.05)
model.fit(X_scaled)
df['Anomaly'] = model.predict(X_scaled)

# --- 4. Create meshgrid for boundary plotting ---
xx, yy = np.meshgrid(
    np.linspace(X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1, 500),
    np.linspace(X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1, 500)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.decision_function(grid)
Z = Z.reshape(xx.shape)

# --- 5. Plot the results ---
plt.figure(figsize=(10, 6))

# Contour: decision function boundary
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

# Plot normal points
normal = df['Anomaly'] == 1
plt.scatter(X_scaled[normal, 0], X_scaled[normal, 1], c='green', s=40, label='Normal')

# Plot anomalies
anomalies = df['Anomaly'] == -1
plt.scatter(X_scaled[anomalies, 0], X_scaled[anomalies, 1], c='red', s=40, label='Anomaly')

plt.xlabel("A (scaled)")
plt.ylabel("B (scaled)")
plt.title("One-Class SVM with Decision Boundary and Anomalies")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
