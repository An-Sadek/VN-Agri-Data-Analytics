# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from plot import ForcastModel

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Dự báo giá nông sản",
    page_icon="🌾",
    layout="wide"
)

# ======================
# CSS STYLING
# ======================
st.markdown("""
<style>
.metric-box {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    text-align: center;
    color: black;
}
.metric-box h4, .metric-box p {
    color: black !important;
}
.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD DATA & MODEL
# ======================
@st.cache_resource
def load_model():
    return ForcastModel(
        model_dir="models",
        csv_path="data/pre_data.csv",
        item_path="data/item.yaml",
        scaler_path="data/scaler.yaml"
    )

@st.cache_data
def load_data():
    df = pd.read_csv("data/pre_data.csv")
    df['Ngày'] = pd.to_datetime(df['Ngày'])
    return df

model = load_model()
df = load_data()

# ======================
# SIDEBAR FILTERS
# ======================
st.sidebar.header("🔧 Cấu hình")

products = sorted(df['Tên_mặt_hàng'].unique())
selected_product = st.sidebar.selectbox("Sản phẩm:", products)

markets = sorted(df[df['Tên_mặt_hàng'] == selected_product]['Thị_trường'].unique())
selected_market = st.sidebar.selectbox("Thị trường:", markets)

price_types = sorted(df[
    (df['Tên_mặt_hàng'] == selected_product) &
    (df['Thị_trường'] == selected_market)
]['Loại_giá'].unique())
selected_price_type = st.sidebar.selectbox("Loại giá:", price_types)

sources = sorted(df[
    (df['Tên_mặt_hàng'] == selected_product) &
    (df['Thị_trường'] == selected_market) &
    (df['Loại_giá'] == selected_price_type)
]['Nguồn'].unique())
selected_source = st.sidebar.selectbox("Nguồn:", sources)

st.sidebar.markdown("---")
selected_models = st.sidebar.multiselect(
    "Mô hình:",
    ["sarimax", "dlm"],
    default=["sarimax"]
)

forecast_days = st.sidebar.slider(
    "Số ngày dự báo:",
    min_value=7,
    max_value=365,
    value=30,
    step=1
)

last_date = df[
    (df['Tên_mặt_hàng'] == selected_product) &
    (df['Thị_trường'] == selected_market) &
    (df['Loại_giá'] == selected_price_type) &
    (df['Nguồn'] == selected_source)
]['Ngày'].max().date()

forecast_start = last_date + timedelta(days=1)
forecast_end = forecast_start + timedelta(days=forecast_days-1)

st.sidebar.write(f"📅 Dữ liệu cuối: {last_date.strftime('%d/%m/%Y')}")
st.sidebar.write(f"🔮 Dự báo từ: {forecast_start.strftime('%d/%m/%Y')}")
st.sidebar.write(f"🔮 Đến: {forecast_end.strftime('%d/%m/%Y')}")

if forecast_days > 90:
    st.sidebar.warning("⚠️ Dự báo > 3 tháng có độ chính xác thấp")

# ======================
# MAIN CONTENT
# ======================
df_filtered = df[
    (df['Tên_mặt_hàng'] == selected_product) &
    (df['Thị_trường'] == selected_market) &
    (df['Loại_giá'] == selected_price_type) &
    (df['Nguồn'] == selected_source)
].sort_values("Ngày")

if df_filtered.empty:
    st.warning("⚠️ Không có dữ liệu!")
    st.stop()

# Lịch sử
st.subheader("📊 Dữ liệu lịch sử")
st.line_chart(df_filtered.set_index("Ngày")["Giá"])

# Dự báo
if selected_models:
    st.subheader("🔮 Dự báo")
    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>Lưu ý:</strong> Dự báo được tính từ ngày cuối dữ liệu. 
    Độ chính xác giảm theo thời gian dự báo.
    </div>
    """, unsafe_allow_html=True)

    forecast_results = []
    for model_type in selected_models:
        features = {
            "Ngày": "",
            "Tên_mặt_hàng": selected_product,
            "Thị_trường": selected_market,
            "Loại_giá": selected_price_type,
            "Nguồn": selected_source,
            "Steps": forecast_days
        }

        y_pred = model.forecast(model_type, features)
        forecast_results.append((model_type, y_pred))

        avg_price = np.mean(y_pred)
        change_pct = ((y_pred[-1] - y_pred[0]) / y_pred[0]) * 100

        st.markdown(f"""
        <div class="metric-box">
            <h4>{model_type.upper()}</h4>
            <p>Dự báo TB: {avg_price:,.0f} VNĐ</p>
            <p>Thay đổi: {change_pct:+.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Plot forecast
        model.plot_forecast(model_type, features)

    # Dataset kết quả
    st.subheader("📋 Kết quả dự báo")
    all_forecast_df = pd.DataFrame()
    for model_type, y_pred in forecast_results:
        dates = [last_date + timedelta(days=i) for i in range(1, len(y_pred)+1)]
        temp_df = pd.DataFrame({
            "Ngày": dates,
            f"{model_type.upper()}": y_pred
        })
        if all_forecast_df.empty:
            all_forecast_df = temp_df
        else:
            all_forecast_df = pd.merge(all_forecast_df, temp_df, on="Ngày")

    st.dataframe(all_forecast_df, use_container_width=True)

    # Download CSV
    csv = all_forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải CSV",
        data=csv,
        file_name=f'du_bao_{selected_product}_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )
else:
    st.info("Chọn ít nhất 1 mô hình để xem dự báo")
