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

        # 显示对话记录
        st.subheader("对话记录")
        
        conversation = [
            ("Human", "使用python做出线性回归，显示点阵图+中心线，使用streamlit做出网页，参数使用拖拉式调整"),
            ("Assistant", 
             "我可以帮您创建一个使用Python的线性回归模型，并用Streamlit制作一个互动式网页来显示结果。这个网页将包含一个散点图、回归线，以及可以用拖拉方式调整的参数。让我们一步一步来实现这个项目。",
             """
             ```python
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
                 # 参数调整
                 n_samples = st.slider("样本数量", 10, 500, 100)
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
                 st.subheader("模型参数:")
                 st.write(f"斜率 (m): {model.coef_[0][0]:.4f}")
                 st.write(f"截距 (b): {model.intercept_[0]:.4f}")
                 
                 st.subheader("模型性能:")
                 st.write(f"均方误差 (MSE): {mse:.4f}")
                 st.write(f"决定系数 (R²): {r2:.4f}")

             if __name__ == "__main__":
                 main()
             ```

             这个Python脚本使用Streamlit创建了一个互动式网页应用，展示了线性回归模型。以下是主要功能：

             1. 使用Streamlit的滑块来调整样本数量和噪音水平。
             2. 根据用户设置的参数生成随机数据。
             3. 使用scikit-learn训练线性回归模型。
             4. 使用Matplotlib绘制静态散点图和回归线。
             5. 创建一个动画GIF，展示散点和回归线如何逐步形成。
             6. 显示模型参数（斜率和截距）以及性能指标（均方误差和R²值）。

             要运行这个应用，您需要安装以下Python库：

             ```
             pip install streamlit numpy pandas matplotlib scikit-learn
             ```

             安装完成后，将代码保存为一个.py文件（例如`linear_regression_app.py`），然后在命令行中运行：

             ```
             streamlit run linear_regression_app.py
             ```

             这将启动一个本地web服务器，并在您的默认浏览器中打开应用。

             您可以通过调整滑块来改变样本数量和噪音水平，应用将实时更新图表、动画和模型参数。

             这个应用提供了一个直观的方式来理解线性回归模型，以及噪音和样本量如何影响模型性能。动画GIF特别有助于理解数据点是如何被添加的，以及回归线是如何适应这些点的。

             如果您需要进一步的解释或者想要对代码进行任何修改，请随时告诉我。
             """),
            ("Human", "新增 一个动画 GIF在静态图下方，展示散点和回归线如何逐步形成"),
            ("Assistant", "好的，我已经按照您的要求更新了代码。现在在静态图下方添加了一个动画GIF，展示散点和回归线如何逐步形成。这个动画可以帮助用户更直观地理解数据点是如何被添加的，以及回归线是如何随着数据点的增加而调整的。\n\n主要的变化包括：\n1. 添加了 `create_animation` 函数来生成动画GIF。\n2. 在主函数中，在显示静态图表后，创建并显示了动画GIF。\n\n这个动画为用户提供了一个更动态的视角来理解线性回归模型的形成过程。您可以调整滑块来改变样本数量和噪音水平，动画会相应地更新，展示不同参数下模型的形成过程。")
        ]

        for i, (role, message, *details) in enumerate(conversation):
            with st.chat_message(role.lower()):
                st.write(f"**{role}**: {message}")
                if details:
                    with st.expander("查看详细代码和解释"):
                        st.markdown(details[0])

    except Exception as e:
        st.error(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
