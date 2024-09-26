import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 設置頁面配置
st.set_page_config(layout="wide", page_title="線性回歸可視化")

# 自定義CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

def generate_data(n_samples, noise, slope, intercept):
    np.random.seed(0)
    X = np.random.rand(n_samples, 1)
    y = intercept + slope * X + np.random.randn(n_samples, 1) * noise
    return X, y

def run_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, X_train, X_test, y_train, y_test

def plot_regression(X, y, model):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.5, label='數據點')
    ax.plot(X, model.predict(X), color='red', label='回歸線')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('線性回歸：散點圖和回歸線')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

st.title('📊 互動式線性回歸可視化')

# 使用 columns 來創建並排的滑塊
col1, col2, col3 = st.columns(3)

with col1:
    n_samples = st.slider('選擇樣本數量', min_value=10, max_value=1000, value=100, step=10)

with col2:
    noise = st.slider('選擇噪音水平', min_value=0.0, max_value=1.0, value=0.1, step=0.05)

with col3:
    slope = st.slider('選擇斜率', min_value=-5.0, max_value=5.0, value=3.0, step=0.1)

intercept = st.slider('選擇截距', min_value=-5.0, max_value=5.0, value=2.0, step=0.1)

X, y = generate_data(n_samples, noise, slope, intercept)
model, mse, r2, X_train, X_test, y_train, y_test = run_linear_regression(X, y)

# 使用 columns 來創建並排的指標
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<p class="big-font">截距</p>', unsafe_allow_html=True)
    st.write(f"{model.intercept_[0]:.4f}")

with col2:
    st.markdown('<p class="big-font">斜率</p>', unsafe_allow_html=True)
    st.write(f"{model.coef_[0][0]:.4f}")

with col3:
    st.markdown('<p class="big-font">均方誤差</p>', unsafe_allow_html=True)
    st.write(f"{mse:.4f}")

with col4:
    st.markdown('<p class="big-font">R²分數</p>', unsafe_allow_html=True)
    st.write(f"{r2:.4f}")

fig = plot_regression(X, y, model)
st.pyplot(fig)

# 添加一個預測部分
st.subheader('🔮 預測')
new_x = st.number_input('輸入一個X值進行預測', value=0.5)
predicted_y = model.predict([[new_x]])[0][0]
st.write(f"對X={new_x}的預測值: {predicted_y:.4f}")
