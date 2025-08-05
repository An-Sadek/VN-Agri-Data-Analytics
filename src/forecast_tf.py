import os

import pandas as pd
import numpy as np
import torch
from model import TimeSeriesTransformer

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 10
MODEL_PATH = "model.pth"
CSV_PATH = "data/scaler_all.csv"

assert os.path.exists(MODEL_PATH)

# --- Load and Preprocess Data ---
df = pd.read_csv(CSV_PATH)
df["Ngày"] = pd.to_datetime(df["Ngày"], format="%Y-%m-%d")
df["dayofyear"] = df["Ngày"].dt.dayofyear
df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

cat_cols = ["Tên_mặt_hàng", "Thị_trường", "Loại_giá", "Nguồn"]
input_cols = cat_cols + ["dayofyear_sin", "dayofyear_cos"]

# Factorize categorical columns
for col in cat_cols:
    df[col] = pd.factorize(df[col])[0]

# --- Load Trained Model ---
model = TimeSeriesTransformer(
    input_dim=len(input_cols),
    model_dim=64,
    num_heads=4,
    num_layers=2,
    output_dim=1
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Reforecast All Training Sequences ---
predictions = []
actuals = []
dates = []
items = []

for item_id in df["Tên_mặt_hàng"].unique():
    item_df = df[df["Tên_mặt_hàng"] == item_id].sort_values("Ngày").reset_index(drop=True)
    item_values = item_df[input_cols + ["Giá"]].values

    for i in range(len(item_values) - SEQ_LEN):
        seq_x = item_values[i:i + SEQ_LEN, :-1]
        target_y = item_values[i + SEQ_LEN, -1]
        target_date = item_df.loc[i + SEQ_LEN, "Ngày"]

        x_input = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_y = model(x_input).item()

        predictions.append(pred_y)
        actuals.append(target_y)
        dates.append(target_date)
        items.append(item_id)

# --- Save Forecast Results ---
results_df = pd.DataFrame({
    "Tên_mặt_hàng": items,
    "Ngày": dates,
    "Actual_Giá": actuals,
    "Forecasted_Giá": predictions
})

results_df.to_csv("forecast_from_training.csv", index=False)
