# plot.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import timedelta
import os
import pickle
import warnings
import yaml
warnings.filterwarnings('ignore')


class ForcastModel:

	def __init__(self, model_dir, csv_path, item_path, scaler_path):
		assert os.path.exists(model_dir)
		assert os.path.exists(csv_path)
		assert os.path.exists(item_path)
		assert os.path.exists(scaler_path)

		self.model_dir = model_dir

		self.df = pd.read_csv(csv_path)

		with open("data/item.yaml", "r", encoding="utf-8") as file:
			self.item_meta = yaml.safe_load(file)

		with open("data/scaler.yaml", "r", encoding="utf-8") as file:
			self.scaler_meta = yaml.safe_load(file)

		self.input_list = ["Ngày", "Tên_mặt_hàng", "Thị_trường", "Loại_giá", "Nguồn"]
		self.cat_cols = ["Tên_mặt_hàng", "Thị_trường", "Loại_giá", "Nguồn"]
		self.exog_cols = ["Thị_trường", "Loại_giá", "Nguồn"]


	def _get_encoded_feature(self, feature_dict: dict):
		encoded_dict = feature_dict.copy()

		# Chuyển các đặc trưng bằng label encoding
		for k in self.cat_cols:
			value = encoded_dict[k]
			encoded_value = self.scaler_meta[k][value]
			encoded_dict[k] = encoded_value

		item = encoded_dict["Tên_mặt_hàng"]

		last_update = self.item_meta[item]["last_update"]
		last_update = pd.to_datetime(last_update)

		# Chuyển ngày về số bước và ngược lại
		if encoded_dict["Ngày"] != "":
			ngay_dudoan = pd.to_datetime(encoded_dict["Ngày"])
			steps = (ngay_dudoan - last_update).days
			assert steps > 0, "Số ngày dự đoán phải lớn hơn 0"
			encoded_dict["Steps"] = steps

		if encoded_dict["Steps"] != 0:
			predict_date = last_update + pd.Timedelta(days=encoded_dict["Steps"])
			encoded_dict["Ngày"] = predict_date

		return encoded_dict

	# Dự đoán
	def forecast(self, model_type: str, feature_dict: dict):
		assert model_type in ["sarimax", "dlm"]
		assert (feature_dict["Ngày"] != "") ^ (feature_dict["Steps"] != 0), "Bắt buộc chỉ có duy nhất Ngày hoặc số bước dự đoán"
		assert  all(k in feature_dict for k in self.input_list), f"Feature dict là từ điển đặc trưng gồm các từ khóa sau: {self.input_list}"

		# Tạo encoded dict tránh việc bị thay đổi trực tiếp
		encoded_dict = self._get_encoded_feature(feature_dict)

		# Lấy idx từ item
		item_idx = encoded_dict["Tên_mặt_hàng"]
		print(item_idx)

		# Chuẩn bị các đặc trưng
		#|-- Lấy số bước
		steps = encoded_dict["Steps"]
		#|-- Chuyển về list
		exog1 = [
			encoded_dict["Thị_trường"],
			encoded_dict["Loại_giá"],
			encoded_dict["Nguồn"]
		]
		exog = []
		for _ in range(steps):
			exog.append(exog1)

		# Đọc model
		with open(f"models/{model_type}/{item_idx}.pkl", "rb") as file:
			model = pickle.load(file)

		# Dự đoán
		y_pred = None
		if model_type == "sarimax":
			y_pred = model.forecast(steps=steps, exog=exog)

		if model_type == "dlm":
			y_pred, _ = model.predictN(N=steps, date=model.n-1, featureDict={"exog": exog})

		return y_pred
	

	def plot_forecast(self, model_type: str, feature_dict: dict, show_historical=True, figsize=(12, 8)):
		# Get encoded features and predictions
		encoded_dict = self._get_encoded_feature(feature_dict)
		
		item_name = feature_dict["Tên_mặt_hàng"]
		item_idx = encoded_dict["Tên_mặt_hàng"]
		item_df = self.df[self.df["Tên_mặt_hàng"] == item_idx].sort_values("Ngày")
		
		# Convert dates
		item_df["Ngày"] = pd.to_datetime(item_df["Ngày"])

		# Forecast
		y_pred = self.forecast(model_type, feature_dict)
		
		# Ensure y_pred is array-like and get its actual length
		y_pred = np.array(y_pred)
		actual_pred_length = len(y_pred)
		
		# Generate future dates for prediction based on actual prediction length
		last_date = item_df["Ngày"].max()
		future_dates = [last_date + timedelta(days=i) for i in range(1, actual_pred_length + 1)]

		# Plotting
		plt.figure(figsize=figsize)

		if show_historical:
			plt.plot(item_df["Ngày"], item_df["Giá"], label="Historical", color="blue")

		plt.plot(future_dates, y_pred, label="Forecast", color="red", linestyle="--")

		plt.title(f"Dự báo giá - {item_name} ({model_type.upper()})")
		plt.xlabel("Ngày")
		plt.ylabel("Giá")
		plt.legend()
		plt.grid(True)

		# Format x-axis dates
		plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
		plt.xticks(rotation=45)

		plt.tight_layout()
		plt.show()



if __name__ == "__main__":
	model = ForcastModel(
		"models",
		"data/pre_data.csv",
		"data/item.yaml",
		"data/scaler.yaml"
	)

	features = {
		"Ngày": "",
		"Tên_mặt_hàng": "Cà phê Robusta nhân xô",
		"Thị_trường": "Đắk Lắk",
		"Loại_giá": "Thương lái thu mua",
		"Nguồn": "CTV địa phương",
		"Steps": 100
	}

	y_pred1 = model.forecast(
		"sarimax",
		feature_dict=features
	)
	print(y_pred1)
	print(np.mean(y_pred1))
	print("\n\n")

	y_pred2 = model.forecast(
		"dlm",
		feature_dict=features
	)
	print(y_pred2)

	model.plot_forecast("dlm", features)

	