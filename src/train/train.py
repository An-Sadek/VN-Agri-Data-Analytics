import os
import pandas as pd
import numpy as np
import pickle
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
warnings.filterwarnings("ignore")

# =======================
# 1. Load & Preprocess Data
# =======================
df = pd.read_csv("data/pre_data.csv")
df["Ngày"] = pd.to_datetime(df["Ngày"])

# Days in data
min_date = df["Ngày"].min()
days = df["Ngày"].apply(lambda x: (x - min_date).days)
max_day = days.max()
df["daysindata_sin"] = np.sin(2 * np.pi * (days / max_day))
df["daysindata_cos"] = np.cos(2 * np.pi * (days / max_day))

# Days in year
days_in_year = df["Ngày"].dt.dayofyear
df["dayinyear_sin"] = np.sin(2 * np.pi * (days_in_year / 365))
df["dayinyear_cos"] = np.cos(2 * np.pi * (days_in_year / 365))

# Feature columns
feature_cols = [
    "Thị_trường", "Loại_giá", "Nguồn", "Ngành_hàng",
    "daysindata_sin", "daysindata_cos", "dayinyear_cos"
]
X = df[feature_cols].values.astype(np.float32)
y = df["Giá"].values.astype(np.float32)


# =======================
# 2. Sequence Creation
# =======================
def create_sequences(X, y, window_size=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)
    return torch.from_numpy(X_seq), torch.from_numpy(y_seq)


window_size = 10
X_seq, y_seq = create_sequences(X, y, window_size)

# Add output dimension
y_seq = y_seq.unsqueeze(1)  # (batch, 1)


# =======================
# 3. DataLoader for batching
# =======================
batch_size = 32
dataset = TensorDataset(X_seq, y_seq)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# =======================
# 4. Transformer Model
# =======================
class TransformerRegressor(nn.Module):
    def __init__(self, input_size, d_model=512, nhead=8, num_layers=6, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True  
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src):
        # src: (batch, seq_len, input_size)
        src = self.input_proj(src)       # (batch, seq_len, d_model)
        out = self.transformer(src, src) # simplified: src as both encoder & decoder
        out = self.fc_out(out[:, -1, :]) # take last time step
        return out


model = TransformerRegressor(input_size=X_seq.shape[2])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# =======================
# 5. Training Loop
# =======================
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
