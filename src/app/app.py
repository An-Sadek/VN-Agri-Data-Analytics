import streamlit as st
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from plot import ForcastModel
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_model():
	"""Load the forecast model with caching"""
	try:
		model = ForcastModel(
			"models",
			"data/pre_data.csv", 
			"data/item.yaml",
			"data/scaler.yaml"
		)
		return model
	except Exception as e:
		st.error(f"Error loading model: {str(e)}")
		return None

def check_data_availability(model, features):
	"""Check if data exists for the selected combination"""
	try:
		# Get the raw data
		df = model.df
		
		# Encode features to match dataset format
		encoded_features = model._get_encoded_feature(features)
		
		# Check if combination exists in dataset
		mask = (
			(df["Tên_mặt_hàng"] == encoded_features["Tên_mặt_hàng"]) &
			(df["Thị_trường"] == encoded_features["Thị_trường"]) &
			(df["Loại_giá"] == encoded_features["Loại_giá"])&
			(df["Nguồn"] == encoded_features["Nguồn"])
		)
		
		matching_data = df[mask]
		
		if len(matching_data) == 0:
			return False, "Không có dữ liệu cho sự kết hợp sản phẩm-vùng-loại giá này"
		else:
			return True, f"Có {len(matching_data)} điểm dữ liệu"
			
	except Exception as e:
		return False, f"Lỗi kiểm tra dữ liệu: {str(e)}"

def display_data_unavailable_message(item_name, market, price_type, source, reason):
	"""Display a user-friendly message when data is not available"""
	st.error("❌ Không thể thực hiện dự báo")
	
	col1, col2 = st.columns([1, 1])
	
	with col1:
		st.warning(f"""
		### 📋 Thông tin yêu cầu:
		- **Sản phẩm:** {item_name}
		- **Thị trường/Vùng:** {market}  
		- **Loại giá:** {price_type}
		- **Nguồn:** {source}
		
		### ⚠️ Vấn đề: 
		{reason} trong cơ sở dữ liệu.
		""")
	
	with col2:
		st.info("""
		### 💡 Gợi ý:
		- Thử chọn thị trường/vùng khác cho sản phẩm này
		- Thử chọn loại giá khác (bán lẻ/bán sỉ/xuất khẩu)
		- Kiểm tra lại tên sản phẩm
		""")

def display_forecast_chart(model, model_type, features, show_historical, chart_height=500):
	"""Display chart using the model's plot_forecast method"""
	try:
		# Use the model's built-in plot_forecast method that returns Plotly figure
		fig = model.plot_forecast(
			model_type=model_type, 
			feature_dict=features, 
			show_historical=show_historical
		)
		
		# Update chart height
		fig.update_layout(height=chart_height)
		
		# Display the Plotly figure in Streamlit
		st.plotly_chart(fig, use_container_width=True)
		
		return True, fig
	except Exception as e:
		st.error(f"Lỗi khi tạo biểu đồ: {str(e)}")
		return False, None

def get_historical_data(model, feature_dict):
	"""Get historical data for the selected item"""
	try:
		encoded_dict = model._get_encoded_feature(feature_dict)
		item_idx = encoded_dict["Tên_mặt_hàng"]
		item_df = model.df[model.df["Tên_mặt_hàng"] == item_idx].sort_values("Ngày")
		item_df["Ngày"] = pd.to_datetime(item_df["Ngày"])
		return item_df
	except Exception:
		return None

def get_available_options(model):
	"""Get available options from scaler metadata"""
	options = {}
	for col in model.cat_cols:
		# Filter out 'max' key and get the actual category values
		col_options = [k for k in model.scaler_meta[col].keys() if k != "max"]
		options[col] = col_options
	return options

def display_forecast_stats(predictions):
	"""Display forecast statistics in a nice format"""
	col1, col2, col3, col4 = st.columns(4)
	
	avg_price = np.mean(predictions)
	min_price = np.min(predictions)
	max_price = np.max(predictions)
	
	with col1:
		st.metric("Giá trung bình", f"{avg_price:,.0f} VNĐ")
	
	with col2:
		st.metric("Giá thấp nhất", f"{min_price:,.0f} VNĐ")
	
	with col3:
		st.metric("Giá cao nhất", f"{max_price:,.0f} VNĐ")
	
	with col4:
		if len(predictions) > 1:
			change = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
			trend_icon = "📈" if change > 0 else "📉" if change < 0 else "➡️"
			st.metric(
				"Xu hướng", 
				f"{trend_icon} {abs(change):.1f}%",
				delta=f"{change:+.1f}%"
			)

def main():
	# Page configuration
	st.set_page_config(
		page_title="Dự báo Giá Nông sản",
		page_icon="📈",
		layout="wide",
		initial_sidebar_state="expanded"
	)
	
	# Header
	st.title("📈 Hệ thống Dự báo Giá Nông sản")
	st.markdown("*Ứng dụng dự báo giá sử dụng mô hình SARIMAX và DLM*")
	st.markdown("---")
	
	# Load model
	model = load_model()
	if model is None:
		st.error("Không thể tải mô hình. Vui lòng kiểm tra các file dữ liệu.")
		st.stop()
	
	# Get available options
	options = get_available_options(model)
	
	# Sidebar for inputs
	with st.sidebar:
		st.header("🔧 Cấu hình Dự báo")
		
		# Model selection
		model_type = st.selectbox(
			"Chọn mô hình dự báo:",
			["sarimax", "dlm"],
			format_func=lambda x: f"{x.upper()} Model",
			help="SARIMAX: Seasonal ARIMA with eXogenous variables\nDLM: Dynamic Linear Model"
		)
		
		st.markdown("### 🛒 Thông tin Sản phẩm")
		
		# Item selection
		item_name = st.selectbox(
			"Tên mặt hàng:",
			options["Tên_mặt_hàng"],
			help="Chọn mặt hàng cần dự báo giá"
		)
		
		# Market selection
		market = st.selectbox(
			"Thị trường:",
			options["Thị_trường"],
			help="Chọn thị trường giao dịch"
		)
		
		# Price type
		price_type = st.selectbox(
			"Loại giá:",
			options["Loại_giá"],
			help="Chọn loại giá (bán lẻ, bán sỉ, etc.)"
		)

		# Price type
		source = st.selectbox(
			"Nguồn:",
			options["Nguồn"],
			help="Chọn nguồn thông tin"
		)
		
		st.markdown("---")
		st.markdown("### 📅 Cấu hình Thời gian")
		
		# Forecast period selection
		forecast_method = st.radio(
			"Phương pháp dự báo:",
			["Số ngày từ hôm nay", "Chọn ngày cụ thể"],
			help="Chọn cách xác định khoảng thời gian dự báo"
		)
		
		steps = 0
		target_date = ""
		
		if forecast_method == "Số ngày từ hôm nay":
			steps = st.number_input(
				"Số ngày dự báo:",
				min_value=1,
				value=30,
				step=1,
				help="Nhập số ngày dự báo tính từ ngày cuối cùng có dữ liệu"
			)
		else:
			min_date = datetime.now().date()
			target_date = st.date_input(
				"Ngày dự báo đến:",
				min_value=min_date,
				value=min_date + timedelta(days=30),
				help="Chọn ngày cuối cùng muốn dự báo"
			)
			target_date = target_date.strftime("%Y-%m-%d")
		
		st.markdown("---")
		st.markdown("### 📊 Tùy chọn Hiển thị")
		
		show_historical = st.checkbox(
			"Hiển thị dữ liệu lịch sử", 
			value=True,
			help="Bật/tắt hiển thị dữ liệu giá lịch sử trên biểu đồ"
		)
		
		# Forecast button
		st.markdown("---")
		forecast_button = st.button(
			"🚀 Thực hiện Dự báo", 
			type="primary",
			use_container_width=True
		)
	
	# Main content area
	if forecast_button:
		try:
			# Prepare feature dictionary
			features = {
				"Ngày": target_date,
				"Tên_mặt_hàng": item_name,
				"Thị_trường": market,
				"Loại_giá": price_type,
				"Nguồn": source,
				"Steps": steps
			}
			
			# Progress indicator
			progress_bar = st.progress(0)
			status_text = st.empty()
			
			# Step 1: Check data availability
			status_text.text("🔍 Đang kiểm tra dữ liệu...")
			progress_bar.progress(20)
			
			data_available, data_message = check_data_availability(model, features)
			
			if not data_available:
				progress_bar.empty()
				status_text.empty()
				display_data_unavailable_message(item_name, market, price_type, source, data_message)
				return
			
			# Step 2: Load data
			status_text.text("📊 Đang tải dữ liệu...")
			progress_bar.progress(40)
			
			# Get historical data if needed
			historical_data = None
			if show_historical:
				historical_data = get_historical_data(model, features)
			
			# Step 3: Run forecast
			status_text.text("🔮 Đang thực hiện dự báo...")
			progress_bar.progress(60)
			
			predictions = model.forecast(model_type, features)
			predictions = np.array(predictions)
			
			# Step 4: Prepare visualization and data
			status_text.text("📈 Đang chuẩn bị biểu đồ...")
			progress_bar.progress(80)
			
			# Generate future dates for the table
			if historical_data is not None and len(historical_data) > 0:
				last_date = historical_data["Ngày"].max()
			else:
				# Use the model's metadata to get the last update date
				encoded_dict = model._get_encoded_feature(features)
				item_idx = encoded_dict["Tên_mặt_hàng"]
				last_date = pd.to_datetime(model.item_meta[item_idx]["last_update"])
			
			future_dates = [last_date + timedelta(days=i) for i in range(1, len(predictions) + 1)]
			
			# Step 5: Complete
			status_text.text("✅ Hoàn thành!")
			progress_bar.progress(100)
			
			# Clear progress indicators
			progress_bar.empty()
			status_text.empty()
			
			# Success message
			st.success(f"✅ Dự báo hoàn thành cho {item_name} sử dụng mô hình {model_type.upper()}!")
			st.info(f"📊 {data_message}")
			
			# Display forecast statistics
			st.subheader("📊 Thống kê Dự báo")
			display_forecast_stats(predictions)
			
			st.markdown("---")
			
			# Create and display chart using model's plot_forecast method
			st.subheader("📈 Biểu đồ Dự báo")
			
			# Display the chart (this is the ONLY plot now)
			chart_success, plotly_fig = display_forecast_chart(
				model, 
				model_type, 
				features, 
				show_historical
			)
			
			st.markdown("---")
			
			# Create columns for data table and summary
			col1, col2 = st.columns([2, 1])
			
			with col1:
				# Forecast table
				st.subheader("📋 Chi tiết Dự báo")
				forecast_df = pd.DataFrame({
					'Ngày': future_dates,
					'Giá dự báo (VNĐ)': [f"{p:,.0f}" for p in predictions],
					'Giá số': predictions
				})
				
				# Display with pagination
				st.dataframe(
					forecast_df[['Ngày', 'Giá dự báo (VNĐ)']],
					use_container_width=True,
					hide_index=True
				)
				
				# Download section
				csv_data = forecast_df[['Ngày', 'Giá số']].copy()
				csv_data.columns = ['Ngay', 'Gia_Du_Bao']
				csv = csv_data.to_csv(index=False, encoding='utf-8')
				
				st.download_button(
					label="📥 Tải xuống dữ liệu dự báo (CSV)",
					data=csv,
					file_name=f"du_bao_{item_name.replace(' ', '_')}_{model_type}_{datetime.now().strftime('%Y%m%d')}.csv",
					mime="text/csv",
					help="Tải xuống kết quả dự báo dưới dạng file CSV"
				)
			
			with col2:
				# Summary information
				st.subheader("ℹ️ Thông tin Tóm tắt")
				
				st.info(f"""
				**Mô hình:** {model_type.upper()}
				
				**Sản phẩm:** {item_name}
				
				**Thị trường:** {market}
				
				**Loại giá:** {price_type}
				
				**Số điểm dự báo:** {len(predictions)} ngày
				
				**Khoảng dự báo:** 
				{future_dates[0].strftime('%d/%m/%Y')} - {future_dates[-1].strftime('%d/%m/%Y')}
				""")
				
				# Price analysis
				if len(predictions) > 1:
					volatility = np.std(predictions)
					st.metric("Độ biến động", f"{volatility:,.0f} VNĐ")
		
		except Exception as e:
			st.error(f"❌ Đã xảy ra lỗi khi thực hiện dự báo:")
			st.exception(e)
			
			with st.expander("🔍 Chi tiết lỗi (dành cho developer)"):
				st.code(str(e))
				st.write("**Gợi ý khắc phục:**")
				st.write("- Kiểm tra các file dữ liệu (CSV, YAML) có tồn tại không")
				st.write("- Đảm bảo mô hình đã được train cho sản phẩm này")
				st.write("- Kiểm tra định dạng ngày tháng")


if __name__ == "__main__":
	main()