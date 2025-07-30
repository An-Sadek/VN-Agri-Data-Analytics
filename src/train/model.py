import os
import torch
import torch.nn as nn
import pandas as pd


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.utils.data import DataLoader, TensorDataset

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.3):
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 500, model_dim))  # assume max seq_len = 500

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoder[:, :x.size(1)]
        x = self.encoder(x)
        out = self.decoder(x[:, -1])  # use last token for prediction
        return out


df = pd.read_csv("data/scaler_all.csv")
df["Ngày"] = pd.to_datetime(df["Ngày"], format="%Y-%m-%d")
cat_cols = ["Tên_mặt_hàng", "Thị_trường", "Loại_giá", "Nguồn"]

SEQ_LEN = 10
input_cols = ["Tên_mặt_hàng", "Thị_trường", "Loại_giá", "Nguồn"]
X, y = [], []

for item in df["Tên_mặt_hàng"].unique():
    item_df = df[df["Tên_mặt_hàng"] == item]

    item_values = item_df[input_cols + ["Giá"]].values

    for i in range(len(item_values) - SEQ_LEN):
        seq_x = item_values[i:i+SEQ_LEN, :-1]  # exclude "Giá"
        seq_y = item_values[i+SEQ_LEN, -1]     # "Giá" tại bước kế tiếp
        X.append(seq_x)
        y.append(seq_y)

X = torch.tensor(X, dtype=torch.float32)  # shape: [batch, seq_len, input_dim]
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# Tạo DataLoader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = TimeSeriesTransformer(
    input_dim=X.shape[2],
    model_dim=64,
    num_heads=4,
    num_layers=2,
    output_dim=1
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1):
    for batch_x, batch_y in loader:
        output = model(batch_x)
        loss = loss_fn(output, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model, "model.pth")
