import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

np.random.seed(42)
N = 2000

rssi = np.random.normal(-60, 10, N)
snr = np.random.normal(25, 7, N)
channel_width = np.random.choice([20, 40, 80], N)
band = np.random.choice([0, 1, 2], N)  # 0=2.4, 1=5, 2=6

df = pd.DataFrame({
    "rssi_dbm": rssi,
    "snr_db": snr,
    "channel_width_mhz": channel_width,
    "band": band
})

def classify_quality(row):
    if row["rssi_dbm"] > -55 and row["snr_db"] > 30:
        return "Excellent"
    elif row["rssi_dbm"] > -65 and row["snr_db"] > 20:
        return "Good"
    elif row["rssi_dbm"] > -75 and row["snr_db"] > 10:
        return "Fair"
    else:
        return "Poor"

df["quality"] = df.apply(classify_quality, axis=1)

df.to_csv("data/synthetic_wifi_data.csv", index=False)

X = df[["rssi_dbm", "snr_db", "channel_width_mhz", "band"]]
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))

joblib.dump(model, "models/wifi_model.pkl")
print("Model saved to models/wifi_model.pkl")
