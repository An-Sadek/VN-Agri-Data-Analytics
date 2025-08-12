import os
import pandas as pd
import numpy as np
import pickle

import warnings

from statsmodels.tsa.api import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv("data/pre_data.csv")

for col in ["Tên_mặt_hàng", "Thị_trường", "Loại_giá", "Nguồn"]:
    lbl_encoder = LabelEncoder()
    df[col] = lbl_encoder.fit_transform(df[col])

for col in ["Tên_mặt_hàng", "Thị_trường", "Loại_giá", "Nguồn"]:
    scaler = MinMaxScaler()  
    df[col] = scaler.fit_transform(df[[col]])

sarimax_results = []

# Tạo thư mục nếu chưa có
os.makedirs("./models/sarimax", exist_ok=True)

for ten_mat_hang, df_mat_hang in df.groupby("Tên_mặt_hàng"):
    for keys, group_df in df_mat_hang.groupby(["Thị_trường", "Loại_giá", "Nguồn"]):
        group_df = group_df.sort_values("Ngày")

        if len(group_df) < 10:
            continue

        y = group_df["Giá"]
        exog = group_df[["Tên_mặt_hàng", "Thị_trường", "Loại_giá", "Nguồn"]]

        try:
            model = SARIMAX(y, exog=exog, order=(1,1,1), seasonal_order=(0,0,0,0))
            model_fit = model.fit(disp=False)

            # Tạo tên file mô hình
            file_name = f"./models/sarimax/{ten_mat_hang}_{keys[0]}_{keys[1]}_{keys[2]}.pkl"
            file_name = file_name.replace("/", "_")  # tránh lỗi nếu tên chứa "/"

            with open(file_name, "wb") as file:
                pickle.dump(model_fit, file)

            forecast = model_fit.forecast(steps=1, exog=exog.tail(1))
            sarimax_results.append(((ten_mat_hang, *keys), forecast.iloc[0]))

        except Exception as e:
            print(f"SARIMAX: Skip group {(ten_mat_hang, *keys)} due to error: {e}")
