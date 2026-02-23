from pydantic import BaseModel

class WifiMetrics(BaseModel):
    rssi_dbm: float
    snr_db: float
    channel_width_mhz: int
    band: int  # 0=2.4GHz, 1=5GHz, 2=6GHz

class PredictionResponse(BaseModel):
    quality: str
    confidence: float
