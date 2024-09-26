import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 顯示標題
st.title('線性迴歸示例')

# 創建模擬數據
np.random.seed(0)
n_samples = 100
X = np.random.rand(n_samples, 1) * 10
y = 2.5 * X + np.random.randn(n_samples, 1) * 2

# 拖動滑桿來設置測試集比例
test_size = st.slider('測試集比例', 0.1, 0.9, 0.2)

# 分割數據為訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 訓練線性迴歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 顯示點陣圖和迴歸線
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color='blue', label='測試數據')
ax.plot(X_test, y_pred, color='red', label='迴歸線')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('線性迴歸')

st.pyplot(fig)

# 顯示迴歸模型的係數與截距
st.write(f'迴歸係數: {model.coef_[0][0]:.2f}')
st.write(f'截距: {model.intercept_[0]:.2f}')

# 顯示你和我的對話記錄
with st.expander('點擊查看完整對話記錄'):
    conversation = '''
    你：使用python做出線性迴歸，顯示點陣圖+中心線，使用streamlit做出網頁，參數使用拖拉式調整(參數設置在圖表上方)，網頁下方顯示我跟你的對話(程式不省略，使用下拉式隱藏or展開)
    ChatGPT：這裡是一個簡單的 Python 示例，使用 Streamlit 來展示線性迴歸圖表，其中包含散點圖、迴歸中心線、拖拉式調整參數的功能，以及網頁下方顯示你與我的對話。...
    '''
    st.write(conversation)

# 提供完整程式碼展開/隱藏功能
with st.expander('點擊查看完整程式碼'):
    code = '''
    import streamlit as st
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    # 顯示標題
    st.title('線性迴歸示例')

    # 創建模擬數據
    np.random.seed(0)
    n_samples = 100
    X = np.random.rand(n_samples, 1) * 10
    y = 2.5 * X + np.random.randn(n_samples, 1) * 2

    # 拖動滑桿來設置測試集比例
    test_size = st.slider('測試集比例', 0.1, 0.9, 0.2)

    # 分割數據為訓練集與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 訓練線性迴歸模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 預測
    y_pred = model.predict(X_test)

    # 顯示點陣圖和迴歸線
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, color='blue', label='測試數據')
    ax.plot(X_test, y_pred, color='red', label='迴歸線')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('線性迴歸')

    st.pyplot(fig)

    # 顯示迴歸模型的係數與截距
    st.write(f'迴歸係數: {model.coef_[0][0]:.2f}')
    st.write(f'截距: {model.intercept_[0]:.2f}')

    # 顯示你和我的對話記錄
    with st.expander('點擊查看完整對話記錄'):
        conversation = '''
        你：使用python做出線性迴歸，顯示點陣圖+中心線，使用streamlit做出網頁，參數使用拖拉式調整(參數設置在圖表上方)，網頁下方顯示我跟你的對話(程式不省略，使用下拉式隱藏or展開)
        ChatGPT：這裡是一個簡單的 Python 示例，使用 Streamlit 來展示線性迴歸圖表，其中包含散點圖、迴歸中心線、拖拉式調整參數的功能，以及網頁下方顯示你與我的對話。...
        '''
        st.write(conversation)

    # 提供完整程式碼展開/隱藏功能
    with st.expander('點擊查看完整程式碼'):
        code = '''
        import streamlit as st
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        '''
        st.code(code)
    '''
    st.code(code)
