import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置页面配置
st.set_page_config(layout="wide", page_title="线性回归可视化")

# 自定义 CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    .question-answer {
        background-color: #e1e1e1;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .code-block {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

def generate_data(n_samples, noise, slope, intercept):
    np.random.seed(0)
    X = np.random.rand(n_samples, 1)
    y = intercept + slope * X + np.random.randn(n_samples, 1) * noise
    return X, y

def run_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, X_train, X_test, y_train, y_test

def plot_regression(X, y, model):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.5, label='数据点')
    ax.plot(X, model.predict(X), color='red', label='回归线')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('线性回归：散点图和回归线')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

st.title('📊 互动式线性回归可视化')

# 使用 columns 来创建并排的滑块
col1, col2, col3 = st.columns(3)

with col1:
    n_samples = st.slider('选择样本数量', min_value=10, max_value=1000, value=100, step=10)

with col2:
    noise = st.slider('选择噪音水平', min_value=0.0, max_value=1.0, value=0.1, step=0.05)

with col3:
    slope = st.slider('选择斜率', min_value=-5.0, max_value=5.0, value=3.0, step=0.1)

intercept = st.slider('选择截距', min_value=-5.0, max_value=5.0, value=2.0, step=0.1)

X, y = generate_data(n_samples, noise, slope, intercept)
model, mse, r2, X_train, X_test, y_train, y_test = run_linear_regression(X, y)

# 使用 columns 来创建并排的指标
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<p class="big-font">截距</p>', unsafe_allow_html=True)
    st.write(f"{model.intercept_[0]:.4f}")

with col2:
    st.markdown('<p class="big-font">斜率</p>', unsafe_allow_html=True)
    st.write(f"{model.coef_[0][0]:.4f}")

with col3:
    st.markdown('<p class="big-font">均方误差</p>', unsafe_allow_html=True)
    st.write(f"{mse:.4f}")

with col4:
    st.markdown('<p class="big-font">R²分数</p>', unsafe_allow_html=True)
    st.write(f"{r2:.4f}")

fig = plot_regression(X, y, model)
st.pyplot(fig)

# 添加一个预测部分
st.subheader('🔮 预测')
new_x = st.number_input('输入一个X值进行预测', value=0.5)
predicted_y = model.predict([[new_x]])[0][0]
st.write(f"对X={new_x}的预测值: {predicted_y:.4f}")

# 问答部分
st.header('❓ 问答部分')
st.markdown('<div class="question-answer">', unsafe_allow_html=True)
st.markdown("**问题1：**")
st.write("使用PYTHON 写出 solve linear regression problem")
st.markdown("**回答1：**")
st.write("好的，我可以为您展示如何使用Python解决线性回归问题。我们将使用NumPy和scikit-learn库来实现这个任务。")

code = """
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 生成样本数据
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1) * 0.1

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"截距: {model.intercept_[0]:.4f}")
print(f"斜率: {model.coef_[0][0]:.4f}")
print(f"均方误差: {mse:.4f}")
print(f"R²分数: {r2:.4f}")

# 使用模型进行预测
new_X = np.array([[0.5]])
predicted_y = model.predict(new_X)
print(f"对X=0.5的预测值: {predicted_y[0][0]:.4f}")
"""

if st.button('显示/隐藏代码'):
    st.code(code, language='python')

st.write("这段代码展示了如何使用Python和scikit-learn库来解决线性回归问题。以下是代码的主要步骤:")
st.write("1. 导入必要的库")
st.write("2. 生成样本数据")
st.write("3. 将数据分割为训练集和测试集")
st.write("4. 创建并训练线性回归模型")
st.write("5. 使用模型进行预测")
st.write("6. 评估模型性能")
st.write("7. 使用训练好的模型进行新的预测")
st.write("这个例子使用了简单的一元线性回归，但相同的方法可以扩展到多元线性回归。")
st.write("您需要安装NumPy和scikit-learn库才能运行这段代码。如果您还没有安装，可以使用以下命令安装:")
st.code("pip install numpy scikit-learn", language='bash')

st.markdown('</div>', unsafe_allow_html=True)
