import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

# Function to generate data
def generate_data(slope, noise_scale, num_points):
    np.random.seed(0)
    x = 2 * np.random.rand(num_points, 1)
    y = 4 + slope * x + noise_scale * np.random.randn(num_points, 1)
    return x, y

# Streamlit sliders for parameters
st.title('作業1-線性迴歸')
slope = st.slider('Slope', 0.0, 10.0, 3.0, 0.1)
noise_scale = st.slider('Noise Scale', 0.0, 5.0, 1.0, 0.1)
num_points = st.slider('Number of Points', 10, 200, 100, 1)

# Generate and plot the data
x, y = generate_data(slope, noise_scale, num_points)
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Plotting the data
fig, ax = plt.subplots()
ax.scatter(x, y, label='Data Points')
ax.plot(x, y_pred, color='red', linewidth=2, label='Regression Line')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Linear Regression')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)
