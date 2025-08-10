import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
torch.manual_seed(42)

df = pd.read_html("data/Ca phe")[0]
df['Ngày'] = pd.to_datetime(df['Ngày'])

del df["Đơn_vị_tính"]
del df["Loại_tiền"]
# --- Fill min->max date
min_date = df['Ngày'].min()
max_date = df['Ngày'].max()
all_dates = pd.date_range(start=min_date, end=max_date, freq='D')

full_df = pd.DataFrame({'Ngày': all_dates})

merged_df = pd.merge(full_df, df, on='Ngày', how='left')

filled_df = merged_df.ffill()
filled_df.to_csv("test/data/full_date_caphe.csv", index=False)

feature_col = ["Thị_trường", "Loại_giá", "Nguồn"]
lbl_df = filled_df.copy()

lbl_encoder = LabelEncoder()
for col in feature_col:
    lbl_df[col] = lbl_encoder.fit_transform(lbl_df[col])


n_market = lbl_df["Thị_trường"].max() + 1
n_price_type = lbl_df["Loại_giá"].max() + 1
n_source = lbl_df["Nguồn"].max() + 1

market_emb = nn.Embedding(n_market, 1)
price_type_emb = nn.Embedding(n_price_type, 1)
source_emb = nn.Embedding(n_source, 1)


# --- Gộp các thuộc tính ---
for date in all_dates:
    date_df = lbl_df[lbl_df["Ngày"] == date]
    
    market_tensor = torch.tensor(date_df["Thị_trường"].to_numpy(), dtype=torch.long)
    price_type_tensor = torch.tensor(date_df["Loại_giá"].to_numpy(), dtype=torch.long)
    source_tensor = torch.tensor(date_df["Nguồn"].to_numpy(), dtype=torch.long)

    market_vec = market_emb(market_tensor)
    price_type_vec = price_type_emb(price_type_tensor)
    source_vec = source_emb(source_tensor)

    # Ghép embedding lại
    combined = torch.cat([market_vec, price_type_vec, source_vec], dim=1)
    print(f"{date}: {combined.shape}")



