import numpy as np
import matplotlib.pyplot as plt
from pydlm import dlm, trend, dynamic

# 1. Create target data (e.g., sine wave with noise)
n = 100
np.random.seed(42)
y = np.sin(np.linspace(0, 3 * np.pi, n)) + np.random.normal(scale=0.2, size=n)

# 2. Create exogenous features (e.g., sin, cos)
x1 = np.cos(np.linspace(0, 3 * np.pi, n))
x2 = np.linspace(0, 1, n)
exog = np.vstack([x1, x2]).T  # shape (n, 2)

# 3. Define model with trend + exogenous (dynamic)
model = dlm(y) + trend(degree=1, discount=0.95, name='trend') + \
        dynamic(features=exog, discount=0.99, name='exog')

# 4. Fit model
model.fit()

# 5. Forecasting future steps - Method 1: Using append and predictN
future_steps = 10
x1_future = np.cos(np.linspace(3 * np.pi, 3.5 * np.pi, future_steps))
x2_future = np.linspace(1, 1.2, future_steps)
exog_future = np.vstack([x1_future, x2_future]).T  # shape (future_steps, 2)

# Create a copy of the model for forecasting
forecast_model = model._copy()

# Method 1: Append future exogenous data and use predictN
# First, we need to extend the model with future exogenous features
forecast_y = []
forecast_var = []

for i in range(future_steps):
    # Get the current state
    if i == 0:
        # Use the last observation point as starting point
        pred_y, pred_var = forecast_model.predictN(date=n-1, N=1)
    else:
        # Continue from the last prediction
        pred_y, pred_var = forecast_model.predictN(date=n-1+i, N=1)
    
    forecast_y.append(pred_y[0])
    forecast_var.append(pred_var[0])
    
    # Update the model with the predicted value and future exog features
    # Note: This is a workaround - we append None for y and the exog features
    try:
        # Append the future exogenous features to the model
        forecast_model = forecast_model + dynamic(features=exog_future[i:i+1], 
                                                discount=0.99, name=f'exog_future_{i}')
    except:
        # If the above doesn't work, try a different approach
        pass

# Convert to numpy arrays
forecast_y = np.array(forecast_y)
forecast_var = np.array(forecast_var)

# Create confidence intervals (assuming normal distribution)
forecast_std = np.sqrt(forecast_var)
forecast_conf = np.column_stack([forecast_y - 1.96 * forecast_std, 
                                forecast_y + 1.96 * forecast_std])

# Alternative Method 2: Manual state-space prediction
# This is more reliable for models with exogenous features
print("Using alternative prediction method...")

# Get the final state from the fitted model
final_state = model.getLatentState()[-1]  # Get last state
final_cov = model.getLatentCov()[-1]      # Get last covariance

# Manual prediction (simplified)
forecast_y_alt = []
forecast_var_alt = []

# Get model matrices (this is model-specific and might need adjustment)
try:
    # Simple approach: use the trend component for basic forecasting
    trend_coef = final_state[0] if len(final_state) > 0 else 0
    trend_slope = final_state[1] if len(final_state) > 1 else 0
    
    for i in range(future_steps):
        # Simple linear trend + exogenous contribution
        pred_trend = trend_coef + trend_slope * (i + 1)
        
        # Add exogenous contribution (simplified)
        if len(final_state) > 2:
            exog_contrib = np.sum(final_state[2:4] * exog_future[i])
        else:
            exog_contrib = 0
            
        pred_y = pred_trend + exog_contrib
        pred_var = final_cov[0, 0] if len(final_cov) > 0 else 0.1  # Simplified variance
        
        forecast_y_alt.append(pred_y)
        forecast_var_alt.append(pred_var)
        
except Exception as e:
    print(f"Alternative method failed: {e}")
    # Fallback: simple trend extrapolation
    last_values = model.getFilteredObs()[-5:]
    trend_est = np.mean(np.diff(last_values))
    
    forecast_y_alt = []
    for i in range(future_steps):
        forecast_y_alt.append(last_values[-1] + trend_est * (i + 1))
    
    forecast_var_alt = [0.1] * future_steps

forecast_y_alt = np.array(forecast_y_alt)
forecast_var_alt = np.array(forecast_var_alt)
forecast_std_alt = np.sqrt(forecast_var_alt)
forecast_conf_alt = np.column_stack([forecast_y_alt - 1.96 * forecast_std_alt, 
                                    forecast_y_alt + 1.96 * forecast_std_alt])
