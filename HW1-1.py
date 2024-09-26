import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.animation as animation
import io

# 設置頁面標題
st.set_page_config(page_title="線性迴歸模型演示")
st.title("線性迴歸模型演示")

# 生成隨機數據
@st.cache_data
def generate_data(n_samples, noise):
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = 2 * X + 1 + np.random.normal(0, noise, (n_samples, 1))
    return X, y

# 創建動畫GIF
def create_animation(X, y, y_pred):
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(min(y.min(), y_pred.min()), max(y.max(), y_pred.max()))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('線性迴歸模型動畫')

        scatter = ax.scatter([], [], color='blue', alpha=0.5)
        line, = ax.plot([], [], color='red', linewidth=2)

        def init():
            scatter.set_offsets(np.empty((0, 2)))
            line.set_data([], [])
            return scatter, line

        def update(frame):
            scatter.set_offsets(np.column_stack((X[:frame], y[:frame])))
            line.set_data(X[:frame], y_pred[:frame])
            return scatter, line

        anim = animation.FuncAnimation(fig, update, frames=min(len(X), 50), init_func=init, blit=True)

        buffer = io.BytesIO()
        anim.save(buffer, writer='pillow', fps=10)
        buffer.seek(0)
        
        plt.close(fig)
        return buffer
    except Exception as e:
        st.error(f"動畫生成錯誤: {str(e)}")
        return None

# 主要的app邏輯
def main():
    try:
        # 側邊欄參數
        st.sidebar.header("模型參數")
        n_samples = st.sidebar.slider("樣本數量", 10, 200, 50)
        noise = st.sidebar.slider("噪音水平", 0.1, 5.0, 1.0)

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
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X, y, color='blue', alpha=0.5)
        ax.plot(X, y_pred, color='red', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('線性迴歸模型')

        # 顯示靜態圖表
        st.pyplot(fig)

        # 創建和顯示動畫GIF
        gif_buffer = create_animation(X, y, y_pred)
        if gif_buffer is not None:
            st.subheader("動畫展示：")
            st.image(gif_buffer, caption="線性迴歸模型動畫", use_column_width=True)

        # 顯示模型參數和性能指標
        st.subheader("模型參數:")
        st.write(f"斜率 (m): {model.coef_[0][0]:.4f}")
        st.write(f"截距 (b): {model.intercept_[0]:.4f}")
        
        st.subheader("模型性能:")
        st.write(f"均方誤差 (MSE): {mse:.4f}")
        st.write(f"決定係數 (R²): {r2:.4f}")

    except Exception as e:
        st.error(f"發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()
