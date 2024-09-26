import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 設置頁面標題
st.set_page_config(page_title="線性迴歸演示")
st.title("線性迴歸演示")

# 創建側邊欄用於參數調整
st.sidebar.header("參數調整")
num_points = st.sidebar.slider("數據點數量", 10, 200, 100)
noise_level = st.sidebar.slider("噪聲級別", 0.0, 2.0, 0.5)
slope = st.sidebar.slider("斜率", -5.0, 5.0, 1.0)
intercept = st.sidebar.slider("截距", -10.0, 10.0, 0.0)

# 生成數據
X = np.linspace(0, 10, num_points).reshape(-1, 1)
y = slope * X.ravel() + intercept + np.random.normal(0, noise_level, num_points)

# 進行線性迴歸
model = LinearRegression()
model.fit(X, y)

# 繪製圖表
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, y, color='blue', alpha=0.5)
ax.plot(X, model.predict(X), color='red', linewidth=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('線性迴歸')

# 在Streamlit中顯示圖表
st.pyplot(fig)

# 顯示迴歸結果
st.write(f"擬合的斜率: {model.coef_[0]:.2f}")
st.write(f"擬合的截距: {model.intercept_:.2f}")

# 添加對話記錄部分
st.header("對話記錄")

conversation = """
Human: 使用python做出線性迴歸，顯示點陣圖+中心線，使用streamlit做出網頁，參數使用拖拉式調整(參數設置在圖表上方)，網頁下方顯示我跟你的對話(程式不省略，使用下拉式隱藏or展開)
