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
        # 參數調整（在圖表上方）
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
            ("學生", "使用python做出線性迴歸，顯示點陣圖+中心線，使用streamlit做出網頁，參數使用拖拉式調整"),
            ("AI", 
             "我可以幫您創建一個使用Python的線性迴歸模型，並用Streamlit製作一個互動式網頁來顯示結果。這個網頁將包含一個散點圖、迴歸線，以及可以用拖拉方式調整的參數。讓我們一步一步來實現這個項目。",
             """
             ```python
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
             ```

             這個Python腳本使用Streamlit創建了一個互動式網頁應用，展示了線性迴歸模型。以下是主要功能：

             1. 使用Streamlit的滑塊來調整樣本數量和噪音水平。
             2. 根據用戶設置的參數生成隨機數據。
             3. 使用scikit-learn訓練線性迴歸模型。
             4. 使用Matplotlib繪製散點圖和迴歸線。
             5. 顯示模型參數（斜率和截距）以及性能指標（均方誤差和R²值）。

             要運行這個應用，您需要安裝以下Python庫：

             ```
             pip install streamlit numpy pandas matplotlib scikit-learn
             ```

             安裝完成後，將代碼保存為一個.py文件（例如`linear_regression_app.py`），然後在命令行中運行：

             ```
             streamlit run linear_regression_app.py
             ```

             這將啟動一個本地web服務器，並在您的默認瀏覽器中打開應用。

             您可以通過調整滑塊來改變樣本數量和噪音水平，應用將實時更新圖表和模型參數。

             這個應用提供了一個直觀的方式來理解線性迴歸模型，以及噪音和樣本量如何影響模型性能。您可以根據需要進一步擴展這個應用，例如添加更多的特徵或不同的回歸算法。

             如果您需要進一步的解釋或者想要對代碼進行任何修改，請隨時告訴我。
             """)
        ]

        for i, (role, message, *details) in enumerate(conversation):
            with st.chat_message(role.lower()):
                st.write(f"**{role}**: {message}")
                if details:
                    with st.expander("查看詳細代碼和解釋"):
                        st.markdown(details[0])

    except Exception as e:
        st.error(f"發生錯誤: {str(e)}")

if __name__ == "__main__":
    main()
