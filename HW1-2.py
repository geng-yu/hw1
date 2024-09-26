import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.colors import Normalize
from matplotlib import cm

# 模擬 Tableau 風格的顏色映射
def plot_regression_tableau_style(X, y, model):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 顏色漸變：從藍色到橙色
    norm = Normalize(vmin=y.min(), vmax=y.max())
    cmap = cm.get_cmap('coolwarm')
    colors = cmap(norm(y.flatten()))

    # 散點圖
    scatter = ax.scatter(X, y, color=colors, alpha=0.7, edgecolor='k', label='數據點')

    # 回歸線
    ax.plot(X, model.predict(X), color='red', linewidth=2, label='回歸線')

    # 添加圖例
    ax.set_xlabel('Sales', fontsize=12)
    ax.set_ylabel('Profit', fontsize=12)
    ax.set_title('Profit vs Sales (Linear Regression)', fontsize=14)
    ax.legend()

    # 模仿 Tableau 風格的坐標網格線
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 添加色條以模仿 Tableau 的顏色標度
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Profit')
    
    return fig

# 設置頁面配置
st.set_page_config(layout="wide", page_title="iOS 風格的線性回歸")

# 生成數據
def generate_data(n_samples, noise, slope, intercept):
    np.random.seed(0)
    X = np.random.rand(n_samples, 1) * 100  # 假設 Sales 的範圍是 0 到 100
    y = intercept + slope * X + np.random.randn(n_samples, 1) * noise
    return X, y

# 線性回歸
def run_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

st.title('📊 Tableau 風格的線性回歸可視化')

# 使用滑塊調整參數
n_samples = st.slider('選擇樣本數量', min_value=10, max_value=1000, value=200, step=10)
noise = st.slider('選擇噪音水平', min_value=0.0, max_value=1.0, value=0.1, step=0.05)
slope = st.slider('選擇斜率', min_value=-5.0, max_value=5.0, value=3.0, step=0.1)
intercept = st.slider('選擇截距', min_value=-5.0, max_value=5.0, value=2.0, step=0.1)

# 生成數據並運行線性回歸
X, y = generate_data(n_samples, noise, slope, intercept)
model, mse, r2 = run_linear_regression(X, y)

# 顯示均方誤差和R²分數
st.write(f"均方誤差 (MSE): {mse:.4f}")
st.write(f"R² 分數: {r2:.4f}")

# 顯示圖表
fig = plot_regression_tableau_style(X, y, model)
st.pyplot(fig)

