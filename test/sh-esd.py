import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyculiarity.detect_ts import detect_ts

# ---- Sample time series data ----
np.random.seed(42)
date_range = pd.date_range(start="2023-01-01", periods=100, freq='D')
data = pd.DataFrame({
    'timestamp': date_range,
    'value': np.sin(np.linspace(0, 10 * np.pi, 100)) + np.random.normal(0, 0.2, 100)
})

# Inject outliers
data.loc[10, 'value'] += 3
data.loc[40, 'value'] -= 4
data.loc[75, 'value'] += 5

# ---- Apply S-H-ESD detection ----
results = detect_ts(data, max_anoms=0.1, direction='both', e_value=True, longterm=False, piecewise_median_period_weeks=0)

# ---- Plotting ----
anoms = pd.DataFrame(results['anoms'])

plt.figure(figsize=(12, 6))
plt.plot(data['timestamp'], data['value'], label='Original Data')
if not anoms.empty:
    plt.scatter(anoms['timestamp'], anoms['anoms'], color='red', label='Anomalies', zorder=5)
plt.title('S-H-ESD Outlier Detection')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
