{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f9486bec-b9b3-46f0-97cb-5f38c93c462b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\User\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\pyplot.py:158: UserWarning: Glyph 32218 (\\N{CJK UNIFIED IDEOGRAPH-7DDA}) missing from current font.\n",
      "  fig.savefig(image, **kwargs)\n",
      "C:\\Users\\User\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\pyplot.py:158: UserWarning: Glyph 24615 (\\N{CJK UNIFIED IDEOGRAPH-6027}) missing from current font.\n",
      "  fig.savefig(image, **kwargs)\n",
      "C:\\Users\\User\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\pyplot.py:158: UserWarning: Glyph 22238 (\\N{CJK UNIFIED IDEOGRAPH-56DE}) missing from current font.\n",
      "  fig.savefig(image, **kwargs)\n",
      "C:\\Users\\User\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\pyplot.py:158: UserWarning: Glyph 27512 (\\N{CJK UNIFIED IDEOGRAPH-6B78}) missing from current font.\n",
      "  fig.savefig(image, **kwargs)\n",
      "C:\\Users\\User\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\pyplot.py:158: UserWarning: Glyph 65306 (\\N{FULLWIDTH COLON}) missing from current font.\n",
      "  fig.savefig(image, **kwargs)\n",
      "C:\\Users\\User\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\pyplot.py:158: UserWarning: Glyph 25955 (\\N{CJK UNIFIED IDEOGRAPH-6563}) missing from current font.\n",
      "  fig.savefig(image, **kwargs)\n",
      "C:\\Users\\User\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\pyplot.py:158: UserWarning: Glyph 40670 (\\N{CJK UNIFIED IDEOGRAPH-9EDE}) missing from current font.\n",
      "  fig.savefig(image, **kwargs)\n",
      "C:\\Users\\User\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\pyplot.py:158: UserWarning: Glyph 22294 (\\N{CJK UNIFIED IDEOGRAPH-5716}) missing from current font.\n",
      "  fig.savefig(image, **kwargs)\n",
      "C:\\Users\\User\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\pyplot.py:158: UserWarning: Glyph 21644 (\\N{CJK UNIFIED IDEOGRAPH-548C}) missing from current font.\n",
      "  fig.savefig(image, **kwargs)\n",
      "C:\\Users\\User\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\pyplot.py:158: UserWarning: Glyph 25976 (\\N{CJK UNIFIED IDEOGRAPH-6578}) missing from current font.\n",
      "  fig.savefig(image, **kwargs)\n",
      "C:\\Users\\User\\anaconda3\\Lib\\site-packages\\streamlit\\elements\\pyplot.py:158: UserWarning: Glyph 25818 (\\N{CJK UNIFIED IDEOGRAPH-64DA}) missing from current font.\n",
      "  fig.savefig(image, **kwargs)\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "\n",
    "def generate_data(n_samples, noise):\n",
    "    np.random.seed(0)\n",
    "    X = np.random.rand(n_samples, 1)\n",
    "    y = 2 + 3 * X + np.random.randn(n_samples, 1) * noise\n",
    "    return X, y\n",
    "\n",
    "def run_linear_regression(X, y):\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "    \n",
    "    model = LinearRegression()\n",
    "    model.fit(X_train, y_train)\n",
    "    \n",
    "    y_pred = model.predict(X_test)\n",
    "    \n",
    "    mse = mean_squared_error(y_test, y_pred)\n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "    \n",
    "    return model, mse, r2, X_train, X_test, y_train, y_test\n",
    "\n",
    "def plot_regression(X, y, model):\n",
    "    fig, ax = plt.subplots(figsize=(10, 6))\n",
    "    ax.scatter(X, y, color='blue', alpha=0.5, label='數據點')\n",
    "    ax.plot(X, model.predict(X), color='red', label='回歸線')\n",
    "    ax.set_xlabel('X')\n",
    "    ax.set_ylabel('y')\n",
    "    ax.set_title('線性回歸：散點圖和回歸線')\n",
    "    ax.legend()\n",
    "    ax.grid(True, linestyle='--', alpha=0.7)\n",
    "    return fig\n",
    "\n",
    "st.title('線性回歸可視化應用')\n",
    "\n",
    "n_samples = st.slider('選擇樣本數量', min_value=10, max_value=1000, value=100, step=10)\n",
    "noise = st.slider('選擇噪音水平', min_value=0.0, max_value=1.0, value=0.1, step=0.05)\n",
    "\n",
    "X, y = generate_data(n_samples, noise)\n",
    "model, mse, r2, X_train, X_test, y_train, y_test = run_linear_regression(X, y)\n",
    "\n",
    "st.write(f\"截距: {model.intercept_[0]:.4f}\")\n",
    "st.write(f\"斜率: {model.coef_[0][0]:.4f}\")\n",
    "st.write(f\"均方誤差: {mse:.4f}\")\n",
    "st.write(f\"R²分數: {r2:.4f}\")\n",
    "\n",
    "fig = plot_regression(X, y, model)\n",
    "st.pyplot(fig)\n",
    "\n",
    "new_x = st.number_input('輸入一個X值進行預測', value=0.5)\n",
    "predicted_y = model.predict([[new_x]])[0][0]\n",
    "st.write(f\"對X={new_x}的預測值: {predicted_y:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23baa6f7-7120-4bf3-befa-2bc1fda887ff",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
