import os
import yaml
from datetime import datetime as dt

import pandas as pd
import numpy as np
import pickle

import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")


class ForecastModel:

    def __init__(self,
        df_path: str,
        models_dir: str,
        item_metadata_path: str,
        scaler_path: str
    ):
        self.df = pd.read_csv(df_path)
        self.items = self.df["Tên_mặt_hàng"].unique().tolist()
        print(self.items)
        self.df["Ngày"] = pd.to_datetime(self.df["Ngày"], format="%Y-%m-%d")

        self.models_dir = models_dir

        with open(item_metadata_path, "r", encoding="utf-8") as file:
            self.item_metadata = yaml.safe_load(file)

        with open(scaler_path, "r", encoding="utf-8") as file:
            self.scaler_metadata = yaml.safe_load(file)


    def forecast_date(self, 
        ngay,
        ten_mat_hang,
        thi_truong,
        loai_gia,
        nguon
    ):
        ngay = pd.to_datetime(ngay, format="%d/%m/%Y")
        item_df = self.df[self.df["Tên_mặt_hàng"] == ten_mat_hang]
        last_update = item_df["Ngày"].max()
        
        steps = abs((last_update - ngay).days)
        item_idx = self.items.index(ten_mat_hang)

        model_path = os.path.join(self.models_dir, f"{item_idx}.pkl")
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        # Exog
        thi_truong =    self.scaler_metadata["Thị_trường"][thi_truong] / \
                        self.scaler_metadata["Thị_trường"]["max"]

        loai_gia =  self.scaler_metadata["Loại_giá"][loai_gia] / \
                    self.scaler_metadata["Loại_giá"]["max"]
        
        nguon = self.scaler_metadata["Nguồn"][nguon] / \
                self.scaler_metadata["Nguồn"]["max"]

        exog = [[thi_truong, loai_gia, nguon]*steps]

        results = model.forecast(exog = exog, steps=steps)

        print(results)


if __name__ == "__main__":
    model = ForecastModel(
        "data/pre_data.csv",
        "models",
        "data/metadata/item.yaml",
        "data/metadata/scaler.yaml"
    )

    model.forecast_date("01/01/2024", "OM 5451", "Cần Thơ", "Thương lái thu mua", "CTV địa phương")