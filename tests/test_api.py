from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_predict():
    payload = {
        "rssi_dbm": -60,
        "snr_db": 25,
        "channel_width_mhz": 20,
        "band": 1
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "quality" in body
    assert "confidence" in body
    assert isinstance(body["quality"], str)
    assert isinstance(body["confidence"], float)
