import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples, noise_level, a, b):
    """生成帶有噪聲的線性數據"""
    x = np.linspace(0, 10, n_samples)
    y = a * x + b + np.random.normal(0, noise_level, n_samples)
    return x, y

def linear_regression(x, y):
    """執行線性回歸"""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    b = (sum_y - a * sum_x) / n
    
    return a, b

def main():
    st.title('交互式線性回歸演示')
    
    # 使用列布局來組織參數控制項
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider('樣本數量', 10, 500, 100)
        noise_level = st.slider('噪聲水平', 0.0, 5.0, 2.0, 0.1)
    with col2:
        true_a = st.slider('真實斜率', -5.0, 5.0, 2.5, 0.1)
        true_b = st.slider('真實截距', -10.0, 10.0, 5.0, 0.1)
    
    x, y = generate_data(n_samples, noise_level, true_a, true_b)
    a, b = linear_regression(x, y)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color='blue', alpha=0.5, label='數據點')
    ax.plot(x, a * x + b, color='red', label='擬合線')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_title(f'線性回歸\n擬合: y = {a:.2f}x + {b:.2f}')
    
    st.pyplot(fig)
    
    # 使用列布局來顯示參數
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"真實參數: a = {true_a:.2f}, b = {true_b:.2f}")
    with col2:
        st.write(f"估計參數: a = {a:.2f}, b = {b:.2f}")
    
    # 添加均方誤差（MSE）的計算和顯示
    mse = np.mean((y - (a * x + b)) ** 2)
    st.write(f"均方誤差 (MSE): {mse:.4f}")

if __name__ == "__main__":
    main()
