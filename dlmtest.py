# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydlm import dlm, trend, seasonality, dynamic
import pickle

# %%
df = pd.read_csv("data/pre_data.csv")

# %%
exog_cols = ["Thị_trường", "Loại_giá", "Nguồn"]

# %%
df.head()

# %%
for item in df["Tên_mặt_hàng"].unique():
    item_df = df[df["Tên_mặt_hàng"] == item]

    y = item_df["Giá"].values.tolist()
    X = item_df[exog_cols].values.tolist()

    model = dlm(y)
    model += dynamic(X, discount=0.95, name="exog")
    model.fit()

    with open(f"results/dlm/{item}.pkl", "wb") as file:
        pickle.dump(model, file)

# %%
h = len(item_df)

# Fix 1: Reshape as a single row with multiple columns
exog_future = [[6, 1, 2]]  # Shape: 1x3 (1 time step, 3 features)

# Alternative Fix 2: Using numpy to ensure proper shape
# exog_future = np.array([[6, 1, 2]])  # Shape: (1, 3)

# Alternative Fix 3: If you need multiple time steps
# exog_future = [[6, 1, 2], [6, 1, 2], [6, 1, 2]]  # Shape: 3x3 (3 time steps, 3 features)

features = {"exog": exog_future}
prediction = model.predict(date=model.n-1, featureDict=features)

# %%
print("Shape of exog_future:", np.array(exog_future).shape)
print("Type of exog_future:", type(exog_future))
print("Type of exog_future[0]:", type(exog_future[0]))
print("Prediction result:", prediction)

# %%
# If you want to make predictions for multiple future periods:
n_periods = 5
exog_future_multi = [[6, 1, 2] for _ in range(n_periods)]  # Shape: 5x3
features_multi = {"exog": exog_future_multi}

# Make predictions for multiple periods
predictions = []
for i in range(n_periods):
    pred = model.predict(date=model.n-1+i, featureDict={"exog": [exog_future_multi[i]]})
    predictions.append(pred)

print("Multi-period predictions:", predictions)