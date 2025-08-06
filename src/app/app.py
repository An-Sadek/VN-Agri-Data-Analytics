# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from plot import (
    create_historical_chart,
    create_forecast_chart,
    create_forecast_dataset, 
    create_forecast_summary,
    check_model_availability,
    check_model_status,
    forecast_with_model
)

# Cấu hình trang
st.set_page_config(
    page_title="Dự báo giá nông sản",
    page_icon="🌾",
    layout="wide"
)

# CSS đơn giản
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

# Tiêu đề
st.title("🌾 Dự báo giá nông sản")
st.markdown("---")

# Load dữ liệu
@st.cache_data
def load_data():
    df_path = "data/pre_data.csv"
    df = pd.read_csv(df_path)
    df['Ngày'] = pd.to_datetime(df['Ngày'])
    return df

df = load_data()

# Sidebar
st.sidebar.header("🔧 Cấu hình")

# Chọn sản phẩm
products = sorted(df['Tên_mặt_hàng'].unique())
selected_product = st.sidebar.selectbox("Sản phẩm:", products)

df_filtered = df[df['Tên_mặt_hàng'] == selected_product]

# Chọn thị trường
markets = sorted(df_filtered['Thị_trường'].unique())
selected_market = st.sidebar.selectbox("Thị trường:", markets)

df_filtered = df_filtered[df_filtered['Thị_trường'] == selected_market]

# Chọn loại giá
price_types = sorted(df_filtered['Loại_giá'].unique())
selected_price_type = st.sidebar.selectbox("Loại giá:", price_types)

df_filtered = df_filtered[df_filtered['Loại_giá'] == selected_price_type]

# Chọn nguồn
sources = sorted(df_filtered['Nguồn'].unique())
selected_source = st.sidebar.selectbox("Nguồn:", sources)

df_filtered = df_filtered[df_filtered['Nguồn'] == selected_source]

# Cấu hình mô hình
st.sidebar.markdown("---")
selected_models = st.sidebar.multiselect(
    "Mô hình:",
    ["SARIMAX", "DLM"],
    default=["SARIMAX"]
)

encoding_type = st.sidebar.radio(
    "Encoding:",
    ["LBL", "OH"]
)

# Cấu hình thời gian dự báo
st.sidebar.markdown("---")
st.sidebar.subheader("Dự báo")

last_date = df_filtered['Ngày'].max().date()
st.sidebar.write(f"📅 Dữ liệu cuối: {last_date.strftime('%d/%m/%Y')}")

# Chỉ cho phép chọn số ngày dự báo (đơn giản hơn)
forecast_days = st.sidebar.slider(
    "Số ngày dự báo:",
    min_value=7,
    max_value=365,
    value=30,
    step=1
)

# Hiển thị ngày dự báo
forecast_start = last_date + timedelta(days=1)
forecast_end = forecast_start + timedelta(days=forecast_days-1)

st.sidebar.write(f"🔮 Dự báo từ: {forecast_start.strftime('%d/%m/%Y')}")
st.sidebar.write(f"🔮 Đến: {forecast_end.strftime('%d/%m/%Y')}")

# Cảnh báo nếu dự báo quá xa
if forecast_days > 90:
    st.sidebar.warning("⚠️ Dự báo > 3 tháng có độ chính xác thấp")

# Main content - 2 biểu đồ riêng biệt
if len(df_filtered) > 0:
    
    # Biểu đồ 1: Dữ liệu lịch sử
    st.subheader("📊 Dữ liệu lịch sử")
    
    fig_historical = create_historical_chart(df_filtered, selected_product)
    if fig_historical is not None:
        st.pyplot(fig_historical)
    else:
        st.warning("Không thể tạo biểu đồ lịch sử!")
    
    # Biểu đồ 2: Dự báo
    st.subheader("🔮 Dự báo")
    
    if selected_models:
        # Thêm disclaimer
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>Lưu ý:</strong> Dự báo được tính từ ngày cuối dữ liệu. 
        Độ chính xác giảm theo thời gian dự báo.
        </div>
        """, unsafe_allow_html=True)
        
        fig_forecast = create_forecast_chart(
            df_filtered, selected_product, selected_models, 
            encoding_type, forecast_days, df
        )
        
        if fig_forecast is not None:
            st.pyplot(fig_forecast)
        else:
            st.warning("Không thể tạo biểu đồ dự báo!")
    else:
        st.info("Chọn ít nhất 1 mô hình để xem dự báo")

else:
    st.warning("⚠️ Không có dữ liệu!")

# Sidebar - Thống kê
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Thống kê")
    
    if len(df_filtered) > 0:
        avg_price = df_filtered['Giá'].mean()
        max_price = df_filtered['Giá'].max()
        min_price = df_filtered['Giá'].min()
        
        st.markdown(f"""
        <div class="metric-box">
            <h4>Giá TB: {avg_price:,.0f} VNĐ</h4>
            <h4>Cao nhất: {max_price:,.0f} VNĐ</h4>
            <h4>Thấp nhất: {min_price:,.0f} VNĐ</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Thống kê dự báo
        for model in selected_models:
            model_status = check_model_status(selected_product, model, encoding_type, df)
            
            if model_status:
                forecast_dates, forecast_prices = forecast_with_model(
                    df_filtered, model, encoding_type, forecast_days, df
                )
                if forecast_prices is not None:
                    summary = create_forecast_summary(forecast_prices, f"{model}_{encoding_type}")
                    if summary:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h4>{model}_{encoding_type}</h4>
                            <p>Dự báo TB: {summary['avg_forecast']:,.0f} VNĐ</p>
                            <p>{summary['trend']} ({summary['change_pct']:+.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)

# Dataset dự báo
st.markdown("---")
st.subheader("Kết quả dự báo")

if len(df_filtered) > 0 and selected_models:
    forecast_df = create_forecast_dataset(
        df_filtered, selected_models, encoding_type, forecast_days,
        selected_product, selected_market, selected_source, selected_price_type, df
    )
    
    if forecast_df is not None:
        # Hiển thị bảng
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        # Download
        col1, col2 = st.columns(2)
        
        with col1:
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải CSV",
                data=csv,
                file_name=f'du_bao_{selected_product}_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
        
        with col2:
            # Thống kê tổng hợp
            all_forecasts = []
            for model in selected_models:
                if check_model_status(selected_product, model, encoding_type, df):
                    _, forecast_prices = forecast_with_model(
                        df_filtered, model, encoding_type, forecast_days, df
                    )
                    if forecast_prices is not None:
                        all_forecasts.extend(forecast_prices)
            
            if all_forecasts:
                avg_all = np.mean(all_forecasts)
                std_all = np.std(all_forecasts)
                
                st.info(f"""
                **Tóm tắt:**
                - Giá TB: {avg_all:,.0f} VNĐ
                - Độ lệch: {std_all:,.0f} VNĐ
                - Số ngày: {forecast_days}
                - Model: {len(selected_models)}
                """)
    else:
        st.warning("Không thể tạo dự báo!")
else:
    st.info("Chọn mô hình để xem dự báo")

# Footer
st.markdown("---")
st.markdown("**Hệ thống dự báo giá nông sản** - Tách biệt dữ liệu lịch sử và dự báo")