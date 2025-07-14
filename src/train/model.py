import os
from datetime import datetime as dt

import pandas as pd
import numpy as np
import pickle

import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")


class ForecastModel:

    def __init__(self, model_path: str, encoder_path: str):
        assert os.path.exists(model_path), "Đường dẫn của model không tồn tại"
        assert os.path.exists(encoder_path), "Đường dẫn của các encoder không tồn tại"

        cols = ["Tên_mặt_hàng", "Thị_trường", "Loại_giá", "Nguồn"]
        self.cols = cols

        # Load model
        with open(f"{model_path}", "rb") as file:
            self.model = pickle.load(file)

        # Tạo từ điển các MinMax Scaler
        self.mm_scaler = dict()
        for col in cols:
            with open(f"{encoder_path}/mm_scaler/{col}.pkl", "rb") as file:
                mm_scaler = pickle.load(file)
                self.mm_scaler.update({col: mm_scaler})

        # Tạo từ điển các Label Scaler
        self.lbl_encoder = dict()
        for col in cols:
            with open(f"{encoder_path}/lbl_scaler/{col}.pkl", "rb") as file:
                mm_scaler = pickle.load(file)
                self.lbl_encoder.update({col: mm_scaler})


    def forecast(self, ten_mat_hang, thi_truong, loai_gia, nguon, steps=1):
        # Encode categorical values
        encoded = dict()
        for col, val in zip(self.cols, [ten_mat_hang, thi_truong, loai_gia, nguon]):
            lbl = self.lbl_encoder[col].transform([val])[0]
            mm = self.mm_scaler[col].transform([[lbl]])[0][0]
            encoded[col] = mm

        exog = pd.DataFrame([encoded]*steps)

        prediction = self.model.forecast(steps=steps, exog=exog)
        return prediction
    

    def forecast_by_date(self, 
        ngay, 
        ten_mat_hang, 
        thi_truong, 
        loai_gia, 
        nguon
    ):
        try:
            ngay = dt.strptime(ngay, "%Y/%m/%d")
        except ValueError as e:
            print("Khong dung dinh dang ngay: %Y/%m/%d")
            print(e)
    
    
    def predict_raw(self, ten_mat_hang, thi_truong, loai_gia, nguon, steps=1):
        exog = np.array([[ten_mat_hang, thi_truong, loai_gia, nguon]] * steps)
        prediction = self.model.forecast(steps=steps, exog=exog)
        return prediction


if __name__ == "__main__":
    model = ForecastModel("src/train/sarimax.pkl", "src/train")
    items = model.lbl_encoder["Tên_mặt_hàng"].classes_ 
    print(items)

    result = model.forecast("Gạo NL 25% tấm", "Kiên Giang", "Thu mua", "CTV địa phương", steps=10)
    print(result)

    raw = model.predict_raw(0, 0, 0, 0, steps=10)
    print(raw)

    model.forecast_by_date("2024-01-01", None, None, None, None)
