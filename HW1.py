import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置页面配置
st.set_page_config(layout="wide", page_title="线性回归可视化")

# 自定义CSS
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
    .user-history {
        background-color: #e1e1e1;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态变量
if 'history' not in st.session_state:
    st.session_state.history = []

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

# 记录用户操作
current_settings = f"样本数: {n_samples}, 噪音: {noise:.2f}, 斜率: {slope:.2f}, 截距: {intercept:.2f}"
if current_settings not in st.session_state.history:
    st.session_state.history.append(current_settings)

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

# 记录预测操作
st.session_state.history.append(f"预测: X={new_x:.2f}, Y={predicted_y:.4f}")

# 添加说明
st.sidebar.header('📘 使用说明')
st.sidebar.write("""
1. 使用滑块调整样本数量、噪音水平、斜率和截距。
2. 观察这些变化如何影响线性回归模型和图表。
3. 在预测部分输入X值，查看模型的预测结果。
""")

# 显示用户交互历史
st.header('👥 用户交互历史')
st.markdown('<div class="user-history">', unsafe_allow_html=True)
for i, action in enumerate(st.session_state.history, 1):
    st.write(f"{i}. {action}")
st.markdown('</div>', unsafe_allow_html=True)

# 添加"如何将代码推送到GitHub"的说明
st.header('🚀 如何将代码推送到GitHub')
st.markdown("""
1. **创建GitHub账户**：如果还没有，在 [GitHub](https://github.com/) 上注册一个账户。

2. **安装Git**：从 [Git官网](https://git-scm.com/downloads) 下载并安装Git。

3. **配置Git**：打开终端，运行以下命令：
   ```
   git config --global user.name "您的名字"
   git config --global user.email "您的邮箱"
   ```

4. **创建新的GitHub仓库**：
   - 登录GitHub
   - 点击右上角的 "+" 图标，选择 "New repository"
   - 填写仓库名称，选择 "Public"
   - 点击 "Create repository"

5. **初始化本地Git仓库**：
   - 打开终端，进入您的项目文件夹
   - 运行 `git init`

6. **添加文件到Git**：
   - 运行 `git add .` 添加所有文件

7. **提交更改**：
   - 运行 `git commit -m "Initial commit"`

8. **链接到GitHub仓库**：
   - 运行 `git remote add origin https://github.com/您的用户名/您的仓库名.git`

9. **推送代码到GitHub**：
   - 运行 `git push -u origin main`

现在您的代码应该已经成功推送到GitHub上了！
""")
