import os
import yaml
from datetime import datetime as dt

import pandas as pd
import numpy as np
import pickle

import warnings

warnings.filterwarnings("ignore")


class ForecastModel:

    def __init__(self, models_path: str, metadata_path: str):
        assert os.path.exists(models_path), f"Đường dẫn models không tồn tại: {models_path}"
        assert os.path.exists(metadata_path), f"Đường dẫn metadata không hợp lệ: {metadata_path}"

        self.models_path = models_path

        with open(os.path.join(metadata_path, "item.yaml"), "r", encoding="utf-8") as file:
            self.item_meta = yaml.safe_load(file)

        with open(os.path.join(metadata_path, "scaler.yaml"), "r", encoding="utf-8") as file:
            self.scaler = yaml.safe_load(file)

    def forecast_by_date(
            self,
            ngay,
            ten_mat_hang,
            thi_truong,
            loai_gia,
            nguon,
            model_type: str = "dlm",
            encoding_type: str = "OH"
    ):
        assert model_type in ["dlm", "sarimax"]
        assert encoding_type in ["OH", "LBL"]

        # Chuyển ngày thành định dạng datetime
        try:
            ngay = pd.to_datetime(ngay, format="%d/%m/%Y")
        except ValueError as e:
            print("Ngày không đúng định dạng dd/mm/YYYY")
            print(e)
            return None

        # Lấy ngày cập nhật cuối và số bước dự báo
        last_update = self.item_meta[ten_mat_hang]["last_update"]
        last_update = pd.to_datetime(last_update, format="%d/%m/%Y")
        steps = abs((ngay - last_update).days)

        # Lấy index từ scaler
        idx = self.scaler["Thị_trường"][thi_truong]

        if encoding_type == "OH":
            # One-hot encoding
            e_thi_truong = np.zeros((self.scaler["Thị_trường"]["max"] + 1), dtype=int)
            e_thi_truong[idx] = 1

            idx = self.scaler["Loại_giá"][loai_gia]
            e_loai_gia = np.zeros((self.scaler["Loại_giá"]["max"] + 1), dtype=int)
            e_loai_gia[idx] = 1

            idx = self.scaler["Nguồn"][nguon]
            e_nguon = np.zeros((self.scaler["Nguồn"]["max"] + 1), dtype=int)
            e_nguon[idx] = 1

            # Kết hợp tất cả
            exog = np.concatenate([e_thi_truong, e_loai_gia, e_nguon])
            exog = np.tile(exog, (steps, 1))
            print("Exog shape:", exog.shape)

        elif encoding_type == "LBL":
            e_thi_truong = self.scaler["Thị_trường"][thi_truong] / self.scaler["Thị_trường"]["max"]
            e_loai_gia = self.scaler["Loại_giá"][loai_gia] / self.scaler["Loại_giá"]["max"]
            e_nguon = self.scaler["Nguồn"][nguon] / self.scaler["Nguồn"]["max"]

            exog = np.array([e_thi_truong, e_loai_gia, e_nguon])
            exog = np.tile(exog, (steps, 1))
            print("Exog shape:", exog.shape)

        # Tải model từ file
        model_path = os.path.join(self.models_path, f"{model_type}_{encoding_type}", f"{idx}.pkl")
        print(f"📦 Loading model from: {model_path}")
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        if model_type == "sarimax":
            y_pred = model.forecast(exog=exog, steps=steps)
            return y_pred

        else:
            print("Chưa hỗ trợ model_type:", model_type)
            return None


if __name__ == "__main__":
    # Dùng đường dẫn tuyệt đối
    models_path = "models"
    metadata_path = "data/metadata"

    model = ForecastModel(models_path, metadata_path)

    y = model.forecast_by_date(
        "01/01/2030",
        "Cà phê Robusta nhân xô",
        "An Giang",
        "Bán buôn",
        "Bán lẻ",
        encoding_type="OH",
        model_type="sarimax"
    )

    if y is not None:
        print("Dự báo:", y)
