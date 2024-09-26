import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# 创建散点图和回归线
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='数据点')
plt.plot(X, model.predict(X), color='red', label='回归线')
plt.xlabel('X')
plt.ylabel('y')
plt.title('线性回归：散点图和回归线')
plt.legend()

# 添加网格线以便更好地查看
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图形
plt.show()
