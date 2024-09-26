import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 設置頁面標題
st.set_page_config(page_title="線性迴歸模型演示", layout="wide")
st.title("線性迴歸模型演示")

# 生成隨機數據
@st.cache_data
def generate_data(n_samples, noise):
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = 2 * X + 1 + np.random.normal(0, noise, (n_samples, 1))
    return X, y

# 主要的app邏輯
def main():
    # 參數調整
    n_samples = st.slider("樣本數量", 10, 500, 100)
    noise = st.slider("噪音水平", 0.1, 5.0, 1.0)

    # 生成數據
    X, y = generate_data(n_samples, noise)

    # 訓練模型
    model = LinearRegression()
    model.fit(X, y)

    # 預測
    y_pred = model.predict(X)

    # 計算模型性能指標
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # 繪製圖表
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.5)
    ax.plot(X, y_pred, color='red', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('線性迴歸模型')

    # 顯示圖表
    st.pyplot(fig)

    # 顯示模型參數和性能指標
    st.subheader("模型參數:")
    st.write(f"斜率 (m): {model.coef_[0][0]:.4f}")
    st.write(f"截距 (b): {model.intercept_[0]:.4f}")

    st.subheader("模型性能:")
    st.write(f"均方誤差 (MSE): {mse:.4f}")
    st.write(f"決定係數 (R²): {r2:.4f}")

if __name__ == "__main__":
    main()
