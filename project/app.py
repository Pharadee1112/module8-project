from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
sgd_model = joblib.load(os.path.join(BASE_DIR, "sgd_model.pkl"))
nn_model = joblib.load(os.path.join(BASE_DIR, "nn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Load data
df = pd.read_csv(os.path.join(BASE_DIR, "bangkok_cleaned.csv"))
df["datetime"] = pd.to_datetime(df["datetime"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():

    date_input = request.form["date"]
    model_type = request.form["model_type"]

    filtered = df[df["datetime"].dt.date ==
                  pd.to_datetime(date_input).date()]

    # 🔥 กัน empty ก่อน
    if filtered.empty or len(filtered) < 4:
        return jsonify({"error": "No sufficient data for selected date."})

# สร้าง target ก่อน
    filtered["target"] = filtered["pm25"].shift(-3)

    # เลือกเฉพาะแถวที่ไม่มี NaN
    filtered = filtered.dropna(subset=["hour", "Temp(C)", "Wind Speed(km/h)", "target"])

    # สร้าง X และ y
    X = filtered[["hour", "Temp(C)", "Wind Speed(km/h)"]]
    y_true = filtered["target"]

    X_scaled = scaler.transform(X)
    
    if model_type == "Stochastic Gradient Descent":
        y_pred = sgd_model.predict(X_scaled)

    elif model_type == "Neural Network (MLP)":
        y_pred = nn_model.predict(X_scaled)

    else:
        return jsonify({"error": "Invalid model type selected."})

    mae = mean_absolute_error(y_true, y_pred)

    # Plot
    plt.figure()
    plt.plot(y_true.values, label="True PM2.5")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title(f"{model_type} | MAE: {mae:.2f}")

    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    plot_path = os.path.join("static", "plot.png")
    plt.savefig(os.path.join(BASE_DIR, plot_path))
    plt.close()

    return jsonify({
        "last_date": date_input,
        "predicted_close": round(float(y_pred[-1]), 2),
        "mae": round(float(mae), 2),
        "mse": 0,
        "plot": "/" + plot_path
    })


if __name__ == "__main__":
    app.run(debug=True)

