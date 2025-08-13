import pandas as pd
import numpy as np

# Example: last historical data point
last_actual = 100  

# Forecast from model
forecast = np.array([105, 108, 110, 113, 115])

# Smoothing factor (0.0 → all forecast, 1.0 → all actual)
alpha = 0.5  

# Blend first forecast point with actual last point
smoothed_forecast = forecast.copy()
smoothed_forecast[0] = alpha * last_actual + (1 - alpha) * forecast[0]

# Optional: gradually blend more points
for i in range(1, len(smoothed_forecast)):
    smoothed_forecast[i] = alpha * smoothed_forecast[i-1] + (1 - alpha) * forecast[i]

print("Original forecast:", forecast)
print("Smoothed forecast:", smoothed_forecast)
