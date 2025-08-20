# plot.py
import numpy as np
import pandas as pd
from datetime import timedelta
import os
import pickle
import warnings
import yaml
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

import sys
sys.path.append("src/train")


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
		"""
		Chuyển từ điển đặc trưng về dạng đã được encode
		"""
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
		"""
		Dự báo giá tương lai bằng mô hình dlm hoặc ARIMAX bằng từ điển đặc trưng
		"""
		assert model_type in ["dlm", "sarimax"]
		assert (feature_dict["Ngày"] != "") ^ (feature_dict["Steps"] != 0), "Bắt buộc chỉ có duy nhất Ngày hoặc số bước dự đoán"
		assert  all(k in feature_dict for k in self.input_list), f"Feature dict là từ điển đặc trưng gồm các từ khóa sau: {self.input_list}"

		# Tạo encoded dict tránh việc bị thay đổi trực tiếp
		encoded_dict = self._get_encoded_feature(feature_dict)

		# Lấy idx từ item
		item_idx = encoded_dict["Tên_mặt_hàng"]

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
		if model_type == "dlm":
			y_pred, _ = model.predictN(N=steps, featureDict={"exog": exog})

		if model_type == "sarimax":
			y_pred = model.forecast(steps=steps, exog=exog)

		return y_pred
	

	def plot_forecast(self, model_type: str, feature_dict: dict, show_historical=True, smoothing_alpha=0.3):
		# Chuyển về dạng label
		encoded_dict = self._get_encoded_feature(feature_dict)
		
		item_name = feature_dict["Tên_mặt_hàng"]
		item_idx = encoded_dict["Tên_mặt_hàng"]
		item_df = self.df[self.df["Tên_mặt_hàng"] == item_idx].sort_values("Ngày")
		
		# Chuyển ngày sang pd date_time
		item_df["Ngày"] = pd.to_datetime(item_df["Ngày"])

		# Dự báo
		y_pred = np.array(self.forecast(model_type, feature_dict))
		actual_pred_length = len(y_pred)

		# Tạo các ngày tương lai
		last_date = item_df["Ngày"].max()
		future_dates = [last_date + timedelta(days=i) for i in range(1, actual_pred_length + 1)]

		# Kết hợp dữ liệu lịch sử + dự báo để smoothing
		full_values = np.concatenate([item_df["Giá"].values, y_pred])
		full_dates = pd.concat([item_df["Ngày"], pd.Series(future_dates)], ignore_index=True)

		# Áp dụng SimpleExpSmoothing
		ses_model = SimpleExpSmoothing(full_values)
		ses_fit = ses_model.fit(smoothing_level=smoothing_alpha, optimized=False)
		smoothed_values = ses_fit.fittedvalues

		# Lấy phần lịch sử và dự báo sau khi smoothing
		smoothed_history = smoothed_values[:len(item_df)]
		smoothed_forecast = smoothed_values[len(item_df):]

		# Create Plotly figure
		fig = go.Figure()

		if show_historical:
			fig.add_trace(go.Scatter(
				x=item_df["Ngày"],
				y=smoothed_history,
				mode="lines",
				name="Historical (smoothed)",
				line=dict(color="blue")
			))

		fig.add_trace(go.Scatter(
			x=future_dates,
			y=smoothed_forecast,
			mode="lines",
			name="Forecast (smoothed)",
			line=dict(color="red", dash="dash")
		))

		fig.update_layout(
			title=f"Dự báo giá - {item_name} ({model_type.upper()} - Smoothed)",
			xaxis_title="Ngày",
			yaxis_title="Giá",
			hovermode="x unified",
			template="plotly_white"
		)

		return fig


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
		"Steps": 10
	}

	# DLM test
	print("DLM")
	y_pred2 = model.forecast(
		"dlm",
		feature_dict=features
	)
	print(y_pred2)

	fig = model.plot_forecast("dlm", features)

	# SARIMA test
	print("SRIMAX")
	y_pred3 = model.forecast(
		"sarimax",
		feature_dict=features
	)
	print(y_pred3)

	model.plot_forecast("dlm", features)
	model.plot_forecast("sarimax", features)

	