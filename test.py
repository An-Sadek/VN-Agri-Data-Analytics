import pandas as pd
from pmdarima import auto_arima

df = pd.read_csv("../../data/pre_data.csv")
df = df.sort_values("Ngày")

for item in df["Tên_mặt_hàng"].unique():
    item_df = df[df["Tên_mặt_hàng"] == item]

    y = item_df["Giá"]
    X = item_df[["Thị_trường", "Loại_giá", "Nguồn"]]

    model = auto_arima(df["Giá"],
                    exogenous=X,  # Include exog variables
                    start_p=0, start_q=0,
                    max_p=10, max_q=10,
                    start_P=0, start_Q=0,
                    max_P=10, max_Q=10,
                    stepwise=True,
                    suppress_warnings=True,
                    D=None,
                    trace=True,
                    error_action='ignore')

