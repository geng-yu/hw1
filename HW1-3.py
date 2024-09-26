import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 設定標題
st.title('線性迴歸展示')

# 創建隨機數據點
n_points = st.slider('選擇數據點數量', 10, 200, 50)
np.random.seed(42)
X = 2 * np.random.rand(n_points, 1)
y = 4 + 3 * X + np.random.randn(n_points, 1)

# 使用線性迴歸模型
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred = lin_reg.predict(X)

# 顯示斜率和截距
st.write(f"模型斜率: {lin_reg.coef_[0][0]:.2f}")
st.write(f"模型截距: {lin_reg.intercept_[0]:.2f}")

# 設置調整噪聲強度的滑桿
noise_level = st.slider('調整噪聲強度', 0.0, 10.0, 1.0)

# 根據噪聲強度重新生成數據
y = 4 + 3 * X + noise_level * np.random.randn(n_points, 1)
y_pred = lin_reg.predict(X)

# 繪製點陣圖與線性迴歸線
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='數據點')
plt.plot(X, y_pred, color='red', label='回歸線')
plt.xlabel('X')
plt.ylabel('y')
plt.title('線性迴歸模型')
plt.legend()

# 在Streamlit上顯示圖表
st.pyplot(plt)

