import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
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

# --- 2. Scale data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['A', 'B']])

# --- 3. Run DBSCAN ---
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X_scaled)
df['Cluster'] = labels

# --- 4. Plot clusters and outliers ---
plt.figure(figsize=(10, 6))

# Plot normal clusters
unique_labels = set(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
for label, color in zip(unique_labels, colors):
    if label == -1:
        # Noise
        plt.scatter(
            X_scaled[labels == label, 0],
            X_scaled[labels == label, 1],
            c='red',
            label='Outliers',
            s=40,
            marker='x'
        )
    else:
        plt.scatter(
            X_scaled[labels == label, 0],
            X_scaled[labels == label, 1],
            c=[color],
            label=f'Cluster {label}',
            s=40
        )

plt.title("DBSCAN Clustering with Outliers")
plt.xlabel("A (scaled)")
plt.ylabel("B (scaled)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
