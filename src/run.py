import os
import yaml
from datetime import datetime as dt
from orbit.forecaster import Forecaster

import pandas as pd
import numpy as np
import pickle

import warnings

warnings.filterwarnings("ignore")


class ForcastModel:

    def __init__(self, models_path: str, metadata_path: str):
        assert os.path.exists(models_path), "Đường dẫn đến models không tồn tại"
        assert os.path.exists(metadata_path), "Đường dẫn đến metadata không hợp lệ"

        self.models_path = models_path

        with open(os.path.join(metadata_path, "item.yaml"), "r", encoding="utf-8") as file:
            self.item_meta = yaml.safe_load(file)

        with open(os.path.join(metadata_path, "scaler.yaml"), "r", encoding="utf-8") as file:
            self.scaler = yaml.safe_load(file)


    def forcast_by_date(
            self,
            ngay,
            ten_mat_hang,
            thi_truong,
            loai_gia,
            nguon,
            model_type: str="dlm",
            encoding_type: str="OH"
    ):
        assert model_type in ["dlm", "sarimax", "dlt"]
        assert encoding_type in ["OH", "LBL"]

        # Chuyển ngày về pd TimeStamp
        try:
            ngay = pd.to_datetime(ngay, format="%d/%m/%Y")
        except ValueError as e:
            print("Không đúng định dạng dd/mm/YYYY")
            print(e)

        # Lấy ngày cập nhật cuối và số bước
        last_update = self.item_meta[ten_mat_hang]["last_update"]
        last_update = pd.to_datetime(last_update, format="%d/%m/%Y")
        steps = abs((ngay-last_update).days)

        # Lấy index mặt hàng từ metadata
        idx = self.scaler["Thị_trường"][thi_truong]

        # Encoding các biến ngoại sinh
        if encoding_type=="OH":
            e_thi_truong = np.zeros(
                (
                    self.scaler["Thị_trường"]["max"] + 1
                ), dtype=int
            )
            e_thi_truong[idx] = 1

            # One-hot Loại giá
            idx = self.scaler["Loại_giá"][loai_gia]
            e_loai_gia = np.zeros(
                (
                    self.scaler["Loại_giá"]["max"] + 1
                ), dtype=int
            )
            e_loai_gia[idx] = 1

            # One-hot nguon
            idx = self.scaler["Nguồn"][nguon]
            e_nguon = np.zeros(
                (
                    self.scaler["Nguồn"]["max"] + 1
                ), dtype=int
            )
            e_nguon[idx] = 1

            # Gộp lại các cột
            exog = np.concat([e_thi_truong, e_loai_gia, e_nguon])
            exog_df = pd.DataFrame(exog, columns=["Thị_trường", "Loại_giá", "Nguồn"])
            exog = np.tile(exog, (steps, 1))

        elif encoding_type=="LBL":
            
            e_thi_truong = self.scaler["Thị_trường"][thi_truong] / \
                            self.scaler["Thị_trường"]["max"]

            e_loai_gia = self.scaler["Loại_giá"][loai_gia] / \
                            self.scaler["Loại_giá"]["max"]
            
            e_nguon = self.scaler["Nguồn"][nguon] / \
                        self.scaler["Nguồn"]["max"]
        
            exog = np.array([e_thi_truong, e_loai_gia, e_nguon])
            exog = np.tile(exog, (steps, 1))

            
        # Lấy model từ idx và loại model
        with open(os.path.join(self.models_path, f"{model_type}_{encoding_type}/{idx}.pkl"), "rb") as file:
            model = pickle.load(file)

        if model_type=="sarimax":
            y_pred = model.forecast(exog=exog, steps=steps)
            print(y_pred)

        if model_type=="dlt":
            forecaster = Forecaster(model=model)
            y_pred = forecaster.predict(df=exog_df, steps_ahead=10)
            print(y_pred)

        
        return y_pred


if __name__ == "__main__":
    model = ForcastModel("models", "data/metadata")
    model.forcast_by_date("01/01/2026", "Cà phê Robusta nhân xô", "An Giang", "Bán buôn", "Bán lẻ", encoding_type="LBL", model_type="sarimax")