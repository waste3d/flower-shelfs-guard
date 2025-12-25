import cv2
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path("models/baseline_full_empty.pkl")

def extract_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    features = [
        np.mean(s),
        np.std(s),
        np.mean(v),
        np.std(v)
    ]
    return np.array(features, dtype=np.float32).reshape(1, -1)

def predict_zone(image_path: str) -> str:
    model = joblib.load(MODEL_PATH)
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read {image_path}")
    
    X = extract_features(img)
    pred = model.predict(X)[0]
    return "full" if pred == 1 else "empty"

if __name__ == "__main__":
    test_path = "data/processed/zones/shelf_full_1_full_zone.jpg"
    label = predict_zone(test_path)
    print(f"{test_path}: {label}")
