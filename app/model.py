import joblib
from pathlib import Path
from typing import Tuple, List

MODEL_PATH = Path("models/wifi_model.pkl")

class WifiModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)

    def predict_quality(self, features: List[float]) -> Tuple[str, float]:
        proba = self.model.predict_proba([features])[0]
        idx = int(proba.argmax())
        label = self.model.classes_[idx]
        confidence = float(proba[idx])
        return label, confidence

wifi_model = WifiModel()
