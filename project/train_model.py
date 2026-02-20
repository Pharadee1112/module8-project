import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor

# =========================
# Load Data
# =========================
df = pd.read_csv("project/bangkok_cleaned.csv")

# Predict ล่วงหน้า 3 ชั่วโมง
df["target"] = df["pm25"].shift(-3)
df = df.dropna()

X = df[["hour", "Temp(C)", "Wind Speed(km/h)"]]
y = df["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =========================
# Scaling
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# =========================
# Model 1: Gradient Descent
# =========================
sgd_model = SGDRegressor(
    max_iter=1000,
    random_state=42
)
sgd_model.fit(X_train_scaled, y_train)

# =========================
# Model 2: Neural Network
# =========================
nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42
)
nn_model.fit(X_train_scaled, y_train)

# =========================
# Save Models
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

joblib.dump(sgd_model, os.path.join(BASE_DIR, "sgd_model.pkl"))
joblib.dump(nn_model, os.path.join(BASE_DIR, "nn_model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

print("Train เสร็จแล้ว (SGD + Neural Network)")