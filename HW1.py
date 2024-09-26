import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.animation as animation
import io

# 设置页面标题
st.set_page_config(page_title="线性回归模型演示", layout="wide")
st.title("线性回归模型演示")

# 生成随机数据
@st.cache_data
def generate_data(n_samples, noise):
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = 2 * X + 1 + np.random.normal(0, noise, (n_samples, 1))
    return X, y

# 创建动画GIF
def create_animation(X, y, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(min(y.min(), y_pred.min()), max(y.max(), y_pred.max()))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('线性回归模型动画')

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

    anim = animation.FuncAnimation(fig, update, frames=len(X), init_func=init, blit=True)

    buffer = io.BytesIO()
    anim.save(buffer, writer='pillow', fps=30)
    buffer.seek(0)
    
    plt.close(fig)
    return buffer

# 主要的app逻辑
def main():
    try:
        # 参数调整（在图表上方）
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("样本数量", 10, 500, 100)
        with col2:
            noise = st.slider("噪音水平", 0.1, 5.0, 1.0)

        # 生成数据
        X, y = generate_data(n_samples, noise)

        # 训练模型
        model = LinearRegression()
        model.fit(X, y)

        # 预测
        y_pred = model.predict(X)

        # 计算模型性能指标
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # 绘制静态图表
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X, y, color='blue', alpha=0.5)
        ax.plot(X, y_pred, color='red', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('线性回归模型')

        # 显示静态图表
        st.pyplot(fig)

        # 创建和显示动画GIF
        gif_buffer = create_animation(X, y, y_pred)
        st.subheader("动画展示：")
        st.image(gif_buffer, caption="线性回归模型动画", use_column_width=True)

        # 显示模型参数和性能指标
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("模型参数:")
            st.write(f"斜率 (m): {model.coef_[0][0]:.4f}")
            st.write(f"截距 (b): {model.intercept_[0]:.4f}")
        
        with col2:
            st.subheader("模型性能:")
            st.write(f"均方误差 (MSE): {mse:.4f}")
            st.write(f"决定系数 (R²): {r2:.4f}")

    except Exception as e:
        st.error(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
