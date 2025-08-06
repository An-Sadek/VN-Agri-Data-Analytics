import os
import yaml
import pickle
import pandas as pd
import numpy as np

class ForecastModel:
    def __init__(self, models_path: str, metadata_path: str):
        self.models_path = models_path
        with open(os.path.join(metadata_path, "item.yaml"), "r", encoding="utf-8") as f:
            self.item_meta = yaml.safe_load(f)
        with open(os.path.join(metadata_path, "scaler.yaml"), "r", encoding="utf-8") as f:
            self.scaler = yaml.safe_load(f)

    def forecast_by_date(self, ngay, ten_mat_hang, thi_truong, loai_gia, nguon, encoding_type="OH"):
        try:
            ngay = pd.to_datetime(ngay, format="%d/%m/%Y")
        except:
            return None

        last_update = pd.to_datetime(self.item_meta[ten_mat_hang]["last_update"], format="%d/%m/%Y")
        steps = abs((ngay - last_update).days)

        if encoding_type == "OH":
            e_tt = np.zeros(self.scaler["Thị_trường"]["max"] + 1)
            e_tt[self.scaler["Thị_trường"][thi_truong]] = 1
            e_lg = np.zeros(self.scaler["Loại_giá"]["max"] + 1)
            e_lg[self.scaler["Loại_giá"][loai_gia]] = 1
            e_n = np.zeros(self.scaler["Nguồn"]["max"] + 1)
            e_n[self.scaler["Nguồn"][nguon]] = 1
            exog = np.tile(np.concatenate([e_tt, e_lg, e_n]), (steps, 1))
        else:
            e_tt = self.scaler["Thị_trường"][thi_truong] / self.scaler["Thị_trường"]["max"]
            e_lg = self.scaler["Loại_giá"][loai_gia] / self.scaler["Loại_giá"]["max"]
            e_n = self.scaler["Nguồn"][nguon] / self.scaler["Nguồn"]["max"]
            exog = np.tile([e_tt, e_lg, e_n], (steps, 1))

        model_idx = self.scaler["Thị_trường"][thi_truong]  # index theo thị trường để map tới model
        model_path = os.path.join(self.models_path, f"sarimax_{encoding_type}", f"{model_idx}.pkl")
        if not os.path.exists(model_path):
            return None

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model.forecast(exog=exog, steps=steps)

if __name__ == "__main__":
    model = ForecastModel(
        "C:/Users/wk/Downloads/VN-Agri-Data-Analytics-train (1)/VN-Agri-Data-Analytics-train/models",
        "C:/Users/wk/Downloads/VN-Agri-Data-Analytics-train (1)/VN-Agri-Data-Analytics-train/data/metadata"
    )

    y = model.forecast_by_date(
        "01/01/2030",
        "Cà phê Robusta nhân xô",
        "An Giang",
        "Bán buôn",
        "Bán lẻ",
        encoding_type="LBL"
    )

    if y is not None:
        print(y)
