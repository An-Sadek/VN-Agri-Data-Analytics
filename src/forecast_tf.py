import pandas as pd
import numpy as np
import torch
from model import TimeSeriesTransformer  # Adjust this if your file is in a different folder

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Constants ---
SEQ_LEN = 10
MODEL_PATH = "model.pth"
CSV_PATH = "data/scaler_all.csv"

# --- Load and Preprocess Data ---
df = pd.read_csv(CSV_PATH)
df["Ngày"] = pd.to_datetime(df["Ngày"], format="%Y-%m-%d")
df["dayofyear"] = df["Ngày"].dt.dayofyear
df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

# Define input columns
cat_cols = ["Tên_mặt_hàng", "Thị_trường", "Loại_giá", "Nguồn"]
input_cols = cat_cols + ["dayofyear_sin", "dayofyear_cos"]

# Encode categorical features (NOTE: This must match training! Use same mappings if possible)
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

# --- Forecast for a Given Product ---
item_id = 0  # Replace with desired product ID (based on factorized "Tên_mặt_hàng")

item_df = df[df["Tên_mặt_hàng"] == item_id].sort_values("Ngày")
if len(item_df) < SEQ_LEN:
    raise ValueError(f"Not enough data for item {item_id} to create a sequence of length {SEQ_LEN}")

latest_seq = item_df[input_cols].values[-SEQ_LEN:]  # shape: (SEQ_LEN, input_dim)
x_input = torch.tensor(latest_seq, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, SEQ_LEN, input_dim)

# --- Make Prediction ---
with torch.no_grad():
    prediction = model(x_input)
    print(f"Forecasted 'Giá' for item {item_id}: {prediction.item():.2f}")
