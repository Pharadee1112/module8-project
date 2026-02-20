import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "nn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

import pandas as pd

sample = pd.DataFrame(
    [[12, 30, 5]],
    columns=["hour", "Temp(C)", "Wind Speed(km/h)"]
)

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("Predicted PM2.5:", prediction[0])