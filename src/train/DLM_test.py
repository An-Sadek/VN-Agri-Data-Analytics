import os
import yaml
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class ForecastModelDLM:
    def __init__(self, models_base_path: str, metadata_path: str, data_path: str):
        self.models_base_path = models_base_path
        
        with open(os.path.join(metadata_path, "item.yaml"), "r", encoding="utf-8") as f:
            self.item_meta = yaml.safe_load(f)
        
        with open(os.path.join(metadata_path, "scaler.yaml"), "r", encoding="utf-8") as f:
            self.scaler = yaml.safe_load(f)
        
        self.data = pd.read_csv(data_path)
        self.data["Ngày"] = pd.to_datetime(self.data["Ngày"], format="%Y-%m-%d")
        self.items = self.data["Tên_mặt_hàng"].unique().tolist()

    def forecast_by_date(self, start_date: str, end_date: str, ten_mat_hang: str, 
                        thi_truong: str, loai_gia: str, nguon: str, encoding_type: str = "LBL"):
        
        start = pd.to_datetime(start_date, format="%d/%m/%Y")
        end = pd.to_datetime(end_date, format="%d/%m/%Y")
        days = (end - start).days + 1
        
        item_idx = self.items.index(ten_mat_hang)
        model_path = os.path.join(self.models_base_path, f"dlm_{encoding_type}", f"{item_idx}.pkl")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        if encoding_type == "LBL":
            exog = np.array([
                [
                    float(self.scaler["Thị_trường"][thi_truong]),
                    float(self.scaler["Loại_giá"][loai_gia]),
                    float(self.scaler["Nguồn"][nguon]),
                ]
                for _ in range(days)
            ], dtype=np.float64)
        else:  # OH
            thị_trường_oh = self.scaler["Thị_trường"][thi_truong]
            loại_giá_oh = self.scaler["Loại_giá"][loai_gia]
            nguồn_oh = self.scaler["Nguồn"][nguon]
            
            if not isinstance(thị_trường_oh, list):
                thị_trường_oh = [float(thị_trường_oh)]
            if not isinstance(loại_giá_oh, list):
                loại_giá_oh = [float(loại_giá_oh)]
            if not isinstance(nguồn_oh, list):
                nguồn_oh = [float(nguồn_oh)]
            
            exog_vector = thị_trường_oh + loại_giá_oh + nguồn_oh
            exog = np.array([exog_vector for _ in range(days)], dtype=np.float64)
        
        predictions = []
        for i in range(days):
            (obs, var) = model.predictN(N=1, date=i)
            predictions.append(obs[0])
        
        for i in range(days):
            day = start + pd.Timedelta(days=i)
            price = float(predictions[i])
            print(f"{day.date()}: {price:.2f}")


if __name__ == "__main__":
    model = ForecastModelDLM(
        models_base_path="C:/Users/wk/Downloads/VN-Agri-Data-Analytics-train (1)/VN-Agri-Data-Analytics-train/models",
        metadata_path="C:/Users/wk/Downloads/VN-Agri-Data-Analytics-train (1)/VN-Agri-Data-Analytics-train/data/metadata",
        data_path="C:/Users/wk/Downloads/VN-Agri-Data-Analytics-train (1)/VN-Agri-Data-Analytics-train/data/pre_data.csv"
    )

    model.forecast_by_date(
        start_date="01/06/2025",
        end_date="07/06/2025",
        ten_mat_hang="Cà phê Robusta nhân xô",
        thi_truong="An Giang",
        loai_gia="Bán lẻ",
        nguon="CTV địa phương",
        encoding_type="OH"
    )
