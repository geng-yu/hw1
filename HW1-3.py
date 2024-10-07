import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 定義更新函數
def update_model(noise_level, slope, num_samples):
    np.random.seed(0)
    X = 2 * np.random.rand(num_samples, 1)
    y = 4 + slope * X + noise_level * np.random.randn(num_samples, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    plt.scatter(X_test, y_test, color='black', label='Actual data')
    plt.plot(X_test, y_pred, color='blue', linewidth=2, label='Predicted line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    st.pyplot(plt.gcf())
    
    st.write(f"Slope: {model.coef_[0][0]}")
    st.write(f"Intercept: {model.intercept_[0]}")

# 創建滑桿
noise_level = st.slider('Noise Level', 0.0, 5.0, 1.0)
slope = st.slider('Slope', 0.0, 10.0, 3.0)
num_samples = st.slider('Num Samples', 10, 500, 100)

update_model(noise_level, slope, num_samples)
