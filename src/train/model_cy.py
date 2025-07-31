import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Transformer Model ---
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.5):
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        out = self.decoder(x[:, -1])  # use the last time step
        return out

# --- Load and Preprocess Data ---
df = pd.read_csv("data/scaler_all.csv")
df["Ngày"] = pd.to_datetime(df["Ngày"], format="%Y-%m-%d")

# Add day-of-year sine as positional encoding
df["dayofyear"] = df["Ngày"].dt.dayofyear
df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

# Define input columns
cat_cols = ["Tên_mặt_hàng", "Thị_trường", "Loại_giá", "Nguồn"]
input_cols = cat_cols + ["dayofyear_sin", "dayofyear_cos"]
SEQ_LEN = 10

# Encode categorical features numerically
for col in cat_cols:
    df[col] = pd.factorize(df[col])[0]

# Build sequences
X, y = [], []
for item in df["Tên_mặt_hàng"].unique():
    item_df = df[df["Tên_mặt_hàng"] == item].sort_values("Ngày")
    item_values = item_df[input_cols + ["Giá"]].values

    for i in range(len(item_values) - SEQ_LEN):
        seq_x = item_values[i:i+SEQ_LEN, :-1]  # input features
        seq_y = item_values[i+SEQ_LEN, -1]     # next step target "Giá"
        X.append(seq_x)
        y.append(seq_y)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# --- DataLoader ---
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- Model ---
model = TimeSeriesTransformer(
    input_dim=X.shape[2],
    model_dim=64,
    num_heads=4,
    num_layers=2,
    output_dim=1
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
loss_history = []

# --- Training ---
EPOCHS = 300
for epoch in range(EPOCHS):
    model.train()
    epoch_losses = []
    for batch_x, batch_y in loader:
        output = model(batch_x)
        loss = loss_fn(output, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# --- Save Model & Loss ---
torch.save(model.state_dict(), "model_state.pth")
pd.DataFrame(loss_history, columns=["Loss"]).to_csv("loss_history.csv", index=False)
