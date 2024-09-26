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
    try:
        # 參數調整（現在在圖表上方）
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("樣本數量", 10, 500, 100)
        with col2:
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

        # 繪製靜態圖表
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, y, color='blue', alpha=0.5)
        ax.plot(X, y_pred, color='red', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('線性迴歸模型')

        # 顯示靜態圖表
        st.pyplot(fig)

        # 顯示模型參數和性能指標
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("模型參數:")
            st.write(f"斜率 (m): {model.coef_[0][0]:.4f}")
            st.write(f"截距 (b): {model.intercept_[0]:.4f}")
        
        with col2:
            st.subheader("模型性能:")
            st.write(f"均方誤差 (MSE): {mse:.4f}")
            st.write(f"決定係數 (R²): {r2:.4f}")

        # 顯示對話記錄
        st.subheader("對話記錄")
        
        conversation = [
            ("Human", "使用python做出線性迴歸，顯示點陣圖+中心線，使用streamlit做出網頁，參數使用拖拉式調整"),
            ("Assistant", "我可以幫您創建一個使用Python的線性迴歸模型，並用Streamlit製作一個互動式網頁來顯示結果。這個網頁將包含一個散點圖、迴歸線，以及可以用拖拉方式調整的參數。讓我們一步一步來實現這個項目。\n\n[此處省略了詳細的代碼和解釋]\n\n如果您需要進一步的解釋或者想要對代碼進行任何修改，請隨時告訴我。")
        ]

        for i, (role, message) in enumerate(conversation):
            with st.chat_message(role.lower()):
                st.write(f"**{role}**: {message}")

    except Exception as e:
        st.error(f"發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()
