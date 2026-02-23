from fastapi import FastAPI
from .schemas import WifiMetrics, PredictionResponse
from .model import wifi_model

app = FastAPI(
    title="WiFi Signal Quality API",
    description="Classifies WiFi signal quality using RF metrics and a trained ML model.",
    version="1.0.0",
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(metrics: WifiMetrics):
    features = [
        metrics.rssi_dbm,
        metrics.snr_db,
        metrics.channel_width_mhz,
        metrics.band,
    ]
    label, confidence = wifi_model.predict_quality(features)
    return PredictionResponse(quality=label, confidence=confidence)
